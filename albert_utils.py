from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import sys,os
sys.path.extend([#'../input/tf2_0_baseline_w_bert/',#'../input/bert_modeling/',
                 '../input/albert/'])
import random
import re

import enum

#import bert_modeling as modeling
#import bert_optimization as optimization
import tokenization
import unicodedata

import numpy as np
import tensorflow as tf

import six

class DummyObject:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

FLAGS=DummyObject(skip_nested_contexts=True,
                 max_position=50,
                 max_contexts=48,
                 max_query_length=64,
                 max_seq_length=512,
                 doc_stride=256,
                 include_unknowns=0.030,
                 n_best_size=5,
                 max_answer_length=30,
                 do_lower_case=True)

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4
  @staticmethod
  def get(val):
    assert val >=0 and val <= 4
    if val == 0:
        return 'UNKNOWN'
    elif val == 1:
        return 'YES'
    elif val == 2:
        return 'NO'
    elif val == 3:
        return 'SHORT'
    elif val == 4:
        return 'LONG'

class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None):
    return super(Answer, cls).__new__(cls, type_, text, offset)


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               example_id,
               qas_id,
               questions,
               doc_tokens,
               doc_tokens_map=None,
               answer=None,
               start_position=None,
               end_position=None):
    self.example_id = example_id
    self.qas_id = qas_id
    self.questions = questions
    self.doc_tokens = doc_tokens
    self.doc_tokens_map = doc_tokens_map
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position

  def __str__(self):
    return '\n'.join([ 'example_id : %s'%self.example_id ,'qas_id : %s'%self.qas_id , 'questions : %s' % self.questions , 'doc_tokens : %s'%self.doc_tokens ,'doc_tokens_map : %s'%self.doc_tokens_map , 'answer : %s'%str(self.answer) , 'start_position : %s'%self.start_position , 'end_position : %s'%self.end_position ])
        

def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)

def should_skip_context(e, idx):
  if (FLAGS.skip_nested_contexts and
      not e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """

  if "annotations" not in e:
      return None, -1, (-1, -1)

  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      idx_list = list()      
      for t_sa in a["short_answers"]:
        idx_list.append(int(t_sa["start_token"]))
        idx_list.append(int(t_sa["end_token"]))
      start_token = min(idx_list)
      end_token = max(idx_list)
      #print('start_token: ' , start_token ,' end_token: ' , end_token) 
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)


def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  #print('get_text_span')
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    #print(i , 'th test_span : ' , t )
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
      #print ('answer : ' , token)
    else:
      pass
      #token = t["token"].replace(" ", "")
      #print ('answer : ' , token)
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "table_html"
  elif first_token == "<P>":
    return "paragraph_html"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "list_html"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "other_html"
  else:
    tf.compat.v1.logging.warning("Unknown candidate type found: %s", first_token)
    return "other_html"


def add_candidate_types_and_positions(e,no_skip_idx):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e,no_skip_idx):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "%s" % context_type


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "content_begin"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e,no_skip_idx):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx) and idx != no_skip_idx:
      #print ( 'should_skip_context : ' , c )
      continue
    #else:
      #print ( 'no should_skip_context : ' , c )
      #pass
    yield idx, c


def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  document_tokens = e["document_text"].split(" ")
  e["document_tokens"] = []
  for idx,token in enumerate(document_tokens):
      #print ( idx , ' th token ' , token )
      e["document_tokens"].append({"token":token if ("<" in token) and (">" in token) else token.lower() , "start_byte":-1, "end_byte":-1, "html_token":("<" in token) and (">" in token)})
      #e["document_tokens"].append({"token":token , "start_byte":-1, "end_byte":-1, "html_token":False)
      #e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":"<" in token})
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)
  #print ('create_example_from_jsonl\'s e: ' , e.keys() )
  add_candidate_types_and_positions(e,annotated_idx)
  #print ('after add_candidate_types_and_positions\'s e: document_tokens' , e['document_tokens'][:20] , 'long_answer_candidates' , e['long_answer_candidates'][:10])
  #print ('annotation : ' , annotation )
  #print ('annotated_idx : ' , annotated_idx )
  #print ('annotated_sa : ' , annotated_sa )
  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
  }

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    idx_list = list()      
    for t_sa in annotation["short_answers"]:
      idx_list.append(int(t_sa["start_token"]))
      idx_list.append(int(t_sa["end_token"]))
    t_start_token = min(idx_list)
    t_end_token = max(idx_list)

    expected_answer_text = get_text_span(
        e, {
            "start_token": t_start_token,
            "end_token": t_end_token,
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])
    
  #print( 'question : ' , question )
  #print( 'annotated_idx : ' , annotated_idx )

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  if annotated_idx >= 0: 
    has_met_annotated_idx = False
  else:
    has_met_annotated_idx = True #for training....

  for idx, _ in candidates_iter(e,annotated_idx):
    if idx == annotated_idx:
      #print ('MET annotated_idx')
      has_met_annotated_idx = True
    elif idx != annotated_idx and len(context_list) >= FLAGS.max_contexts:
      #print ('annotated_idx : ' , annotated_idx )
      #print ('NOT MET annotated_idx')
      continue
    #print ('idx : ' ,idx)
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts and has_met_annotated_idx:
      #print ('MAX_CONTEXT')
      break
  
  #print ( 'context_idxs len: ' , len(context_idxs))
  #print ( 'context_idxs : ' , context_idxs)
  #print ( 'context_list len: ' , len(context_list))
  #print ( 'context_list : ' , context_list)
  
  if "document_title" not in e:
      e["document_title"] = e["example_id"]

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
  }

  #print ('example : ' , example )

  single_map = []
  single_context = []
  offset = 0
  for context in context_list:
    #single_map.extend([-1, -1])
    #single_context.append("%s" % (context["id"], context["type"]))
    single_map.extend([-1])
    single_context.append("%s" %context["type"])
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1
  
  #print('single_map : ' , single_map )
  #print('single_context : ' , single_context )

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]
    #print( 'annotated_idx : ' , annotated_idx)
    #print( 'expected : ' , expected)
    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])
  #print ( 'example : ' , example )
  return example


def make_nq_answer(contexts, answer):
  """Makes an Answer object following NQ conventions.

  Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

  Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]

  if (answer["candidate_id"] == -1 or start >= len(contexts) or
      end > len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "long":
    answer_type = AnswerType.LONG
  else:
    answer_type = AnswerType.SHORT

  return Answer(answer_type, text=contexts[start:end], offset=start)


def read_nq_entry(entry, is_training):
  """Converts a NQ entry into a list of NqExamples."""

  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  examples = []
  contexts_id = entry["id"]
  contexts = entry["contexts"]
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in contexts:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  questions = []
  for i, question in enumerate(entry["questions"]):
    qas_id = "{}".format(contexts_id)
    question_text = question["input_text"]
    #print (i , 'th question' , question_text )
    start_position = None
    end_position = None
    answer = None
    if is_training:
      answer_dict = entry["answers"][i]
      answer = make_nq_answer(contexts, answer_dict)

      # For now, only handle extractive, yes, and no.
      if answer is None or answer.offset is None:
        #print ('NO ANSWER')
        continue
      start_position = char_to_word_offset[answer.offset]
      end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

      # Only add answers where the text can be exactly recovered from the
      # document. If this CAN'T happen it's likely due to weird Unicode
      # stuff so we will just skip the example.
      #
      # Note that this means for training mode, every example is NOT
      # guaranteed to be preserved.
      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
      cleaned_answer_text = " ".join(
          tokenization.whitespace_tokenize(answer.text))
      if actual_text.find(cleaned_answer_text) == -1:
        #tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
        #                   cleaned_answer_text)
        print ("Could not find answer: '%s' vs. '%s'", actual_text,
                           cleaned_answer_text)
        continue

    questions.append(question_text)
    #print('AAA',contexts_id )
    #print('BBB',qas_id )
    #print('CCC',questions[:] )
    #print('DDD',doc_tokens )
    #print('EEE',entry.get("contexts_map", None) )
    #print('FFF', answer)
    #print('GGG', start_position)
    #print('HHH', end_position)

    example = NqExample(
        example_id=int(contexts_id),
        qas_id=qas_id,
        questions=questions[:],
        doc_tokens=doc_tokens,
        doc_tokens_map=entry.get("contexts_map", None),
        answer=answer,
        start_position=start_position,
        end_position=end_position)
    examples.append(example)
  return examples

class ConvertExamples2Features:
    def __init__(self,tokenizer, is_training, output_fn, target_json=None ,collect_stat=False):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.output_fn = output_fn
        self.num_spans_to_ids = collections.defaultdict(list) if collect_stat else None
        self.target_dict = None
        if is_training:
            self.target_dict = read_target_value(target_json)
        
    def __call__(self,example):
        example_index = example.example_id
        features = convert_single_example(example, self.tokenizer, self.target_dict , self.is_training)
        if self.num_spans_to_ids is not None:
            num_spans_to_ids[len(features)].append(example.qas_id)
        #print( 'features num : ' , len(features))
        for feature in features:
            #print('----------------------------------------------')
            #print(str(feature))
            #print('----------------------------------------------')
            feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.doc_span_index
            self.output_fn(feature)
        return len(features)

def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def convert_single_example(example, tokenizer, target_dict,is_training):
  """Converts a single NqExample into a list of InputFeatures."""
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  for (i, token) in enumerate(example.doc_tokens):
    #print ( i , ' th ' , token )
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenization.encode_ids( tokenizer.sp_model, tokenization.preprocess_text(token , lower=FLAGS.do_lower_case))
    #print ( i , ' th sub ' , sub_tokens)
    if len(sub_tokens) == 0:
      #print('NO SUB TOKEN')
      sub_tokens = [tokenizer.sp_model.PieceToId("<unk>")]

    assert len(sub_tokens) != 0
    tok_to_orig_index.extend([i] * len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)

  #print ( 'orig_to_tok_index : ' , orig_to_tok_index )
  #print ( 'tok_to_orig_index : ' , tok_to_orig_index )
  #print ( 'all_doc_tokens : ', all_doc_tokens )
  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  if example.doc_tokens_map:
    tok_to_orig_index = [
        example.doc_tokens_map[index] for index in tok_to_orig_index
    ]
  #print ( 'after tok_to_orig_index : ' , tok_to_orig_index )
  # QUERY
  query_tokens = []

  query_tokens.extend(tokenization.encode_ids(tokenizer.sp_model,tokenization.preprocess_text( example.questions[-1], lower=FLAGS.do_lower_case)))
  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[-FLAGS.max_query_length:]

  # ANSWER
  tok_start_position = 0
  tok_end_position = 0
  if is_training:
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
      tok_end_position = len(all_doc_tokens) - 1

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length","contain_answer"])
  doc_spans = []
  start_offset = 0
  center_start_offset = -1
  len_of_all_doc_tokens = len(all_doc_tokens)
  '''
  if is_training: #when training mode run , add center doc_span
    if tok_start_position != 0 and tok_end_position != 0:
     while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      length = min(length, max_tokens_for_doc)
      doc_start = start_offset
      doc_end = start_offset + length - 1
      contains_an_annotation = ( tok_start_position > doc_start and tok_end_position < doc_end)
      if contains_an_annotation:
        doc_spans.append(_DocSpan(start=start_offset, length=length , contain_answer=True))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, 128)
    
    rnd_cnt = 0
    answer_span_cnt = 3 if len(doc_spans) == 0 and random.random() >= 0.5 else 1
    if answer_span_cnt != 0:
      for idx in range (100):
        random_start_offset = 0
        if len_of_all_doc_tokens-max_tokens_for_doc > 1:
          random_start_offset = random.randrange(0,len_of_all_doc_tokens-max_tokens_for_doc)
        elif len(doc_spans) > 0:
          break
        
        if random_start_offset >= 0:
          #print ( 'temp_start_offset : ' , left_start_offset)
          #print ( 'random_start_offset : ' , random_start_offset)
          length = len_of_all_doc_tokens - random_start_offset
          length = min(length, max_tokens_for_doc)
          doc_start = random_start_offset
          doc_end = random_start_offset + length - 1
          contains_an_annotation = ( tok_start_position > doc_start and tok_end_position < doc_end)
          if not contains_an_annotation:
             doc_spans.append(_DocSpan(start=random_start_offset, length=length , contain_answer=False))
             rnd_cnt += 1
          if rnd_cnt >= answer_span_cnt:
            break
  else:
  '''
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)

    doc_start = start_offset
    doc_end = start_offset + length - 1
    contains_an_annotation = False
    if example.answer != None and example.answer.type != AnswerType.UNKNOWN:
      contains_an_annotation = ( tok_start_position >= doc_start and tok_end_position <= doc_end)
    if contains_an_annotation:
      doc_spans.append(_DocSpan(start=start_offset, length=length , contain_answer=True))
    else:
      doc_spans.append(_DocSpan(start=start_offset, length=length , contain_answer=False))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  #print (doc_spans)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      #is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
      token_is_max_context[len(tokens)] = False # is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokens

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (FLAGS.max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)
    
    #print ( 'tokens : ' , tokens )
    #print ( 'input_ids : ' , input_ids )  
    #print ( 'input_mask : ' , input_mask )
    #print ( 'segment_ids : ' , segment_ids )

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = False
      contains_part_of_annotation = False
      if example.answer.type != AnswerType.UNKNOWN:
        contains_an_annotation = ( tok_start_position >= doc_start and tok_end_position <= doc_end )
        contains_part_of_annotation = (doc_start <= tok_start_position and tok_start_position <= doc_end ) or (doc_start <= tok_end_position and tok_end_position <= doc_end )
      '''
      if ((not contains_an_annotation) or
          example.answer.type == AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
      #  if (FLAGS.include_unknowns < 0 or
      #      random.random() > FLAGS.include_unknowns ):
        if not doc_span.include_trainset or example.answer.type != AnswerType.UNKNOWN:
            continue
      ''' 
      if (not contains_an_annotation ) or example.answer.type == AnswerType.UNKNOWN:
        #if random.random() > FLAGS.include_unknowns or contains_part_of_annotation: # random.random() > (0.036764706):
        #  continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type
      
      t_dict = target_dict[example.example_id]
      target_yes_no = t_dict['yes_no_answer'][0]
      target_short_answers = t_dict['short_answers']
      target_long_answer = t_dict['long_answer']
      target_answer_type = 0
      if contains_an_annotation:
          if target_yes_no == 'YES':
            target_answer_type = 1
          elif target_yes_no == 'NO':
            target_answer_type = 2
          elif len(target_short_answers) > 0:
            target_answer_type = 3
          elif len(target_long_answer) > 0:
            target_answer_type = 4
      
          if target_answer_type != answer_type:
            #print ( target_dict )
            print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(tok_to_orig_index[doc_start] , tok_to_orig_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%(tok_to_orig_index[tok_start_position] , tok_to_orig_index[tok_end_position] ))
            print ('ANSWER_ERROR : %s target_answer_type != answer_type : %s != %s'%(example.example_id , target_answer_type , answer_type))
            #print ('ANSWER_ERROR : %s target_answer_type != answer_type : %s BBBBB %s'%(example.example_id , tokens , token_to_orig_map ) )
          elif answer_type != AnswerType.UNKNOWN:
            if target_answer_type == 3:
                has_answer = False
                positions_list = list()
                for positions_tup in target_short_answers:
                    positions_list.append(positions_tup[0])
                    positions_list.append(positions_tup[1])
                
                if token_to_orig_map[start_position] == min(positions_list) and token_to_orig_map[end_position]+1 == max(positions_list):
                    has_answer = True
                if not has_answer:
                    print ( 'token_to_orig_map:%s'%token_to_orig_map )
                    print ( 'answer_text:%s'%answer_text )
                    print ( 'start_position:%s , end_position:%s'%(start_position , end_position) )
                    #print ( target_dict )
                    print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(tok_to_orig_index[doc_start] , tok_to_orig_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%(tok_to_orig_index[tok_start_position] , tok_to_orig_index[tok_end_position] ))
                    print ('ANSWER_ERROR: %s %s type : %s != %s'%(example.example_id , AnswerType.get(target_answer_type) , (token_to_orig_map[start_position] , token_to_orig_map[end_position]) , target_short_answers ))
                    #print ('ANSWER_ERROR: %s %s type : %s BBBBB %s'%(example.example_id , AnswerType.get(target_answer_type) , tokens , token_to_orig_map ) )
            elif target_answer_type == 4:
                has_answer = False
                for positions_tup in target_long_answer:
                    if ( token_to_orig_map[start_position] - positions_tup[0] >= 0 or token_to_orig_map[start_position] - positions_tup[0] <= 5 ) and ( positions_tup[1] - token_to_orig_map[end_position] > 0 or positions_tup[1] - token_to_orig_map[end_position] <= 5  ):
                        has_answer = True
                if not has_answer:
                    #print ( target_dict )
                    print ( 'token_to_orig_map:%s'%token_to_orig_map )
                    print ( 'answer_text:%s'%answer_text )
                    print ( 'start_position:%s , end_position:%s'%(start_position , end_position) )
                    print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(tok_to_orig_index[doc_start] , tok_to_orig_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%(tok_to_orig_index[tok_start_position] , tok_to_orig_index[tok_end_position] ))
                    print ('ANSWER_ERROR: %s %s type : %s != %s'%(example.example_id , AnswerType.get(target_answer_type) , (token_to_orig_map[start_position] , token_to_orig_map[end_position]) , target_long_answer ))
                    #print ('ANSWER_ERROR: %s %s type : %s BBBBB %s'%(example.example_id , AnswerType.get(target_answer_type) , tokens , token_to_orig_map ) )
        
      #answer_text = " ".join(tokens[start_position:(end_position + 1)])
    #print ('making for %s , %s ' % ( example.example_id , doc_span ))
    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=token_to_orig_map,
        token_is_max_context=token_is_max_context,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)

    features.append(feature)

  return features


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  tokenize_fn = tokenizer.tokenize
  if apply_basic_tokenization:
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
    if _SPECIAL_TOKENS_RE.match(token):
      if token in tokenizer.vocab:
        tokens.append(token)
      else:
        tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
    else:
      tokens.extend(tokenize_fn(token))
  return tokens

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type

  def __str__(self):
    return '\n'.join([ 'unique_id : %s'%self.unique_id ,'example_index : %s'%self.example_index , 'doc_span_index : %s' % self.doc_span_index , 'tokens : %s'%self.tokens ,'token_to_orig_map : %s'%self.token_to_orig_map , 'token_is_max_context : %s'%self.token_is_max_context , 'input_ids : %s'%self.input_ids , 'input_mask : %s'%self.input_mask , 'segment_ids : %s'%self.segment_ids , 'start_position : %s'%self.start_position , 'end_position : %s'%self.end_position , 'answer_text : %s'%self.answer_text , 'answer_type : %s'%self.answer_type ])


def file_iter(input_file, tqdm=None):
  """Read a NQ json file into a list of NqExample."""
  input_paths = tf.io.gfile.glob(input_file)
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
    else:
      return tf.io.gfile.GFile(path, "r")

  for path in input_paths:
    tf.compat.v1.logging.info("Reading: %s", path)
    with _open(path) as input_file:
      if tqdm is not None:
        input_file = tqdm(input_file)
      for line in input_file:
        yield line
    
def nq_examples_iter(input_file, is_training,tqdm=None):
  """Read a NQ json file into a list of NqExample."""
  #print ('start func nq_examples_iter.....') 
  input_paths = tf.io.gfile.glob(input_file)
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
    else:
      return tf.io.gfile.GFile(path, "r")

  for path in input_paths:
    #tf.compat.v1.logging.info
    #print("Reading: %s"% path)
    with _open(path) as input_file:
      if tqdm is not None:
        input_file = tqdm(input_file)
      for index, line in enumerate(input_file):
        #print ( index , ' th line')
        #print ( 'line : ' , line)
        entry = create_example_from_jsonl(line)
        #print ( 'entry : '  , entry )
        yield read_nq_entry(entry, is_training)
   
RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_id"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    token_map = [-1] * len(feature.input_ids)
    for k, v in feature.token_to_orig_map.items():
      token_map[k] = v
    features["token_map"] = create_int_feature(token_map)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def read_candidates_from_one_split(input_path):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  print("Reading examples from: %s" % input_path)
  if input_path.endswith(".gz"):
    with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
        
  else:
    with tf.io.gfile.GFile(input_path, "r") as input_file:
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
  return candidates_dict


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.io.gfile.glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    #print ( input_path )
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict

def read_target_value(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.io.gfile.glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    with tf.io.gfile.GFile(input_path, "r") as input_file:
      for index, line in enumerate(input_file):
        e = json.loads(line)
        
        if "annotations" not in e:
            continue

        annotations = e["annotations"]
        e_id = e["example_id"]
        t_s_answer_tup_list = list()
        t_l_answer_tup_list = list()
        t_yes_no_answer_list  = list()
        for a in annotations:
            t_short_answer_list = a['short_answers']
            if len(t_short_answer_list ) > 0:
                for t_t_dict in t_short_answer_list:
                    t_start_idx = t_t_dict['start_token']
                    t_end_idx = t_t_dict['end_token']
                    if t_start_idx != -1 and t_end_idx != -1:
                        t_s_answer_tup_list.append( (t_start_idx , t_end_idx))
            
            t_long_answer_dict = a['long_answer']
            t_start_idx = t_long_answer_dict['start_token']
            t_end_idx = t_long_answer_dict['end_token']
            if t_start_idx != -1 and t_end_idx != -1:
                t_l_answer_tup_list.append( (t_start_idx , t_end_idx))
            t_yes_no_answer_str = a['yes_no_answer']
            assert t_yes_no_answer_str in ("YES", "NO", "NONE")
            t_yes_no_answer_list.append(t_yes_no_answer_str)
            final_dict[e_id] = { 'long_answer' : t_l_answer_tup_list , 'short_answers' : t_s_answer_tup_list , 'yes_no_answer' : t_yes_no_answer_list }
  return final_dict

Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])

class EvalExample(object):
  """Eval data available for a single example."""
  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}

  def __str__(self):
    return '\n'.join([ 'example_id : %s'%self.example_id ,'candidates : %s'%len(self.candidates) , 'results : %s' % len(self.results) , 'features : %s'% len(self.features) ] )


class ScoreSummary(object):
  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


ResultSpan = collections.namedtuple("ResultSpan", ["start_token_idx", "end_token_idx", "score"])

def top_k_indices(logits,n_best_size,token_map):
    indices = np.argsort(logits[1:])+1
    indices = indices[token_map[indices]!=-1]
    return indices[-n_best_size:]

def remove_duplicates(span):
    start_end = []
    for s in span:
        cont = 0
        if not start_end:
            start_end.append(ResultSpan(s[0], s[1], s[2] ))
            cont += 1
        else:
            for i in range(len(start_end)):
                if start_end[i][0] == s[0] and start_end[i][1] == s[1]:
                    cont += 1
        if cont == 0:
            start_end.append(ResultSpan(s[0], s[1], s[2] ))
            
    return start_end

def get_short_long_span(predictions, example):
    
    sorted_predictions = sorted(predictions, reverse=True)
    short_span = []
    long_span = []
    for prediction in sorted_predictions:
        score, _, summary, start_span, end_span   = prediction
        # get scores > zero
        #if score > 0:
        short_span.append(ResultSpan(int(start_span), int(end_span), float(score) ))

    short_span = remove_duplicates(short_span)

    for s in range(len(short_span)):
        for c in example.candidates:
            start = short_span[s].start_token_idx
            end = short_span[s].end_token_idx
            ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span.append(ResultSpan(int(c["start_token"]), int(c["end_token"]), float(short_span[s].score) ))
                break
    long_span = remove_duplicates(long_span)
    
    if not long_span:
        long_span = []
    if not short_span:
        short_span = []
        
    return short_span, long_span
'''
def get_short_long_span(predictions, example):
    
    #sorted_predictions = sorted(predictions, reverse=True)
    short_span = []
    long_span = []
    start_span_dict = dict()
    end_span_dict = dict()
    start_cand_dict = dict()
    end_cand_dict = dict()
    cand_list = [ 0 for _t in range(len(example.candidates))]
    for prediction in predictions:
        score, _, summary, start_span, end_span   = prediction
        # get scores > zero
        #if score > 0:
        #short_span.append(ResultSpan(int(start_span), int(end_span), float(score) ))
        start_span = int(start_span)
        end_span = int(end_span)
        if start_span not in start_span_dict.keys():
          start_span_dict[start_span] = 1
        else:
          start_span_dict[start_span] += 1

        if end_span not in end_span_dict.keys():
          end_span_dict[end_span] = 1
        else:
          end_span_dict[end_span] += 1
        
        for idx,c in enumerate(example.candidates):
            ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
            if c["top_level"] and c["start_token"] <= start_span and c["end_token"] >= end_span:
                start_cand_dict[start_span] = idx
                end_cand_dict[end_span] = idx
                cand_list[idx] += 1
                break

    #print ( 'example.example_id : ' ,int(example.example_id) )
    #print ( 'start_span_dict : ' , start_span_dict )
    #print ( 'end_span_dict : ' , end_span_dict ) 
    #print ( 'cand_list : ' , cand_list ) 

    for prediction in predictions:
        score, _, summary, start_span, end_span   = prediction
        start_span = int(start_span)
        end_span = int(end_span)
        score = float(score) + (start_span_dict[start_span] - 1) * 0.25 + (end_span_dict[end_span] - 1) * 0.25 + ( (cand_list[start_cand_dict[start_span]]-1)*0.41 if start_span in start_cand_dict.keys() and end_span in end_cand_dict else 0 )
        short_span.append(ResultSpan(int(start_span), int(end_span), score  ))
    
    short_span = sorted(short_span ,reverse=True , key=lambda key: key[2])
    #short_span = remove_duplicates(short_span)
    
    for s in range(len(short_span)):
        for c in example.candidates:
            start = short_span[s].start_token_idx
            end = short_span[s].end_token_idx
            ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span.append(ResultSpan(int(c["start_token"]), int(c["end_token"]), float(short_span[s].score) ))
                break
    long_span = remove_duplicates(long_span)
    
    if not long_span:
        long_span = []
    if not short_span:
        short_span = []
        
    return short_span, long_span

def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = FLAGS.n_best_size
    max_answer_length = FLAGS.max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)
        if len(start_indexes)==0:
            continue
        end_indexes   = top_k_indices(result.end_logits,n_best_size,token_map)
        if len(end_indexes)==0:
            continue
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]

        answer_type = int(np.argmax(result.answer_type_logits))
        start_logit = int(np.argmax(result.start_logits))
        end_logit = int(np.argmax(result.end_logits))
        if start_logit == 0 and end_logit == 0 and answer_type == 0:
          continue
        
        temp_prediction = list()
        for _, (start_index,end_index) in enumerate(indexes):  
            summary = ScoreSummary()
            
            summary.short_span_score = (
                result.start_logits[start_index] +
                result.end_logits[end_index])
            summary.cls_token_score = ( result.start_logits[0] + result.end_logits[0] )
            #summary.cls_token_score = ( result.start_logits[0] + result.end_logits[0] + result.answer_type_logits[0]  )*2./3.
            summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1
            
            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            temp_prediction.append( (score, i, summary, start_span, end_span ) )

            i += 1 # to break ties
        sorted_predictions = sorted(temp_prediction, reverse=True)
        for row in sorted_predictions[:min(len(sorted_predictions),1)]:
          predictions.append( row )
    # Default empty prediction.
    #score = -10000.0
    short_span = []
    long_span  = []
    summary    = ScoreSummary()

    if predictions:
        short_span, long_span = get_short_long_span(predictions, example)
      
    summary.predicted_label = {
        "example_id": int(example.example_id),
        "long_answers": {
          "tokens_and_score": long_span,
          #"end_token": long_span,
          "start_byte": -1,
          "end_byte": -1
        },
        #"long_answer_score": answer_score,
        "short_answers": {
          "tokens_and_score": short_span,
          #"end_token": short_span,
          "start_byte": -1,
          "end_byte": -1,
          "yes_no_answer": "NONE"
        },
        #"short_answer_score": answer_scores,
        
        #"answer_type_logits": summary.answer_type_logits.tolist(),
        #"answer_type": int(np.argmax(summary.answer_type_logits))
       }

    return summary

'''
def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = FLAGS.n_best_size
    max_answer_length = FLAGS.max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)
        if len(start_indexes)==0:
            continue
        end_indexes   = top_k_indices(result.end_logits,n_best_size,token_map)
        if len(end_indexes)==0:
            continue
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]
        for _, (start_index,end_index) in enumerate(indexes):  
            summary = ScoreSummary()
            
            summary.short_span_score = (
                result.start_logits[start_index] +
                result.end_logits[end_index])
            summary.cls_token_score = ( result.start_logits[0] + result.end_logits[0] )
            #summary.cls_token_score = ( result.start_logits[0] + result.end_logits[0] + result.answer_type_logits[0]  )*2./3.
            summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1

            answer_type = int(np.argmax(summary.answer_type_logits))
            start_logit = int(np.argmax(result.start_logits))
            end_logit = int(np.argmax(result.end_logits))
            
            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            if start_logit != 0 and end_logit != 0 and answer_type != 0:
              predictions.append( (score, i, summary, start_span, end_span ) )
            
            #summary.short_span_score = (
            ##    result.start_logits[start_index] +
            #    result.end_logits[end_index])/2.
            #summary.cls_token_score = (
            #    result.start_logits[0] + result.end_logits[0])/2.
            #summary.answer_type_logits = result.answer_type_logits #-result.answer_type_logits.mean()
            #start_span = token_map[start_index]
            #end_span = token_map[end_index] + 1
            #answer_type = int(np.argmax(summary.answer_type_logits))
            #start_logit = int(np.argmax(result.start_logits))
            #end_logit = int(np.argmax(result.end_logits))
            # Span logits minus the cls logits seems to be close to the best.
            #score = summary.short_span_score # - summary.cls_token_score
            #if start_logit != 0 and end_logit != 0 and answer_type != 0:
            #  predictions.append((score, i, summary, start_span, end_span,summary.answer_type_logits))
            
            i += 1 # to break ties

    # Default empty prediction.
    #score = -10000.0
    short_span = []
    long_span  = []
    summary    = ScoreSummary()

    if predictions:
        short_span, long_span = get_short_long_span(predictions, example)
      
    summary.predicted_label = {
        "example_id": int(example.example_id),
        "long_answers": {
          "tokens_and_score": long_span,
          #"end_token": long_span,
          "start_byte": -1,
          "end_byte": -1
        },
        #"long_answer_score": answer_score,
        "short_answers": {
          "tokens_and_score": short_span,
          #"end_token": short_span,
          "start_byte": -1,
          "end_byte": -1,
          "yes_no_answer": "NONE"
        },
        #"short_answer_score": answer_scores,
        
        #"answer_type_logits": summary.answer_type_logits.tolist(),
        #"answer_type": int(np.argmax(summary.answer_type_logits))
       }

    return summary

def compute_eval_dict(candidates_dict, dev_features, raw_results,targets_dict,tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res.unique_id),1, res) for res in raw_results]

    examples_by_id = [(int(k),0,v) for k, v in candidates_dict.items()]
  
    features_by_id = [(int(d['unique_id']),2,d) for d in dev_features] 
  
    # Join examples with features and raw results.
    examples = []
    #print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    #print('done.')
    
    for idx, type_, datum in merged:
        #print (idx,type_)
        if type_==0: #isinstance(datum, list):
            #print ( idx )
            examples.append(EvalExample(idx, datum))
        elif type_==2: #"token_map" in datum:
            examples[-1].features[idx] = datum
            #print( datum['token_map'][datum['start_positions']] )
            #print( datum['token_map'][datum['end_positions']]+1 )
            #print( AnswerType.get(datum['answer_types']) )
        else:
            examples[-1].results[idx] = datum
    
    # Construct prediction objects.
    #print('Computing predictions...')
   
    nq_pred_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        #print(str(e))
        if len(e.features) == 0 or len (e.results) == 0:
            continue
        assert len(e.features) == len(e.results)
        summary = compute_predictions(e)
        
        target_result_dict = targets_dict[e.example_id]
        summary.predicted_label['target_short_answers'] = target_result_dict['short_answers']
        summary.predicted_label['target_long_answer'] = target_result_dict['long_answer']
        summary.predicted_label['target_yes_no_answer'] = target_result_dict['yes_no_answer']

        nq_pred_dict[e.example_id] = summary.predicted_label

    return nq_pred_dict

def compute_pred_dict(candidates_dict, dev_features, raw_results,tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res.unique_id),1, res) for res in raw_results]

    examples_by_id = [(int(k),0,v) for k, v in candidates_dict.items()]
  
    features_by_id = [(int(d['unique_id']),2,d) for d in dev_features] 
  
    # Join examples with features and raw results.
    examples = []
    #print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    #print('done.')
    
    for idx, type_, datum in merged:
        #print (idx,type_)
        if type_==0: #isinstance(datum, list):
            examples.append(EvalExample(idx, datum))
        elif type_==2: #"token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum
    
    # Construct prediction objects.
    #print('Computing predictions...')
   
    nq_pred_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        #print(str(e))
        #if len(e.features) == 0 or len (e.results) == 0:
        #    continue
        #assert len(e.features) == len(e.results)
        
        summary = compute_predictions(e)
        nq_pred_dict[e.example_id] = summary.predicted_label

    return nq_pred_dict


