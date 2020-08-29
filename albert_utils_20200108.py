'''
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
                 doc_stride=128,
                 include_unknowns=1.0,
                 n_best_size=20,
                 max_answer_length=30,
                 do_lower_case=True )

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
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
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
    if not t["skip_token"]:
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
    if not t["skip_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "htmltable"
  elif first_token == "<P>":
    return "htmlp"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "htmlu"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "htmlt"
  else:
    tf.compat.v1.logging.warning("Unknown candidate type found: %s", first_token)
    return "htmlo"


def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "%s=%d" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "!"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      #print ( 'should_skip_context : ' , c )
      continue
    else:
      #print ( 'no should_skip_context : ' , c )
      pass
    yield idx, c


def create_example_from_jsonl(line):
  #print (line)
  #line = unicodedata.normalize('NFKD', line)
  #print ('after' ,  line)
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  #print ( 'before : ' ,  e["document_text"] )
  #e["document_text"] = unicodedata.normalize('NFKD', e["document_text"])
  #print ( 'after : ' ,  e["document_text"] )
  #print ( 'after : ' ,  e["document_text"].encode('utf-8-sig').decode("utf-8-sig") )

  document_tokens = e["document_text"].split(" ")
  e["document_tokens"] = []
  for idx,token in enumerate(document_tokens):
      #print ( idx , ' th token ' , token )
      if not isinstance(token, str):
        print ('NOT STR TOKEN', token)
      if isinstance(token, bytes):
        print ('BYTES TOKEN', token)
      #if '\ufeff' in token or '\u200e' in token or '\u200b' in token or 'u200d' in token:
      #  print ('SPECIAL TOKEN', token)

#      def is_special_char(c):
#        return ( ord(c) >= 0x2000 and ord(c) <= 0x200f  ) or ( ord(c) >= 0xfe00 and ord(c) <= 0xfe0f ) or (ord(c) >= 0x2028 and ord(c) <= 0x202f ) or (  ord(c) >= 0x2060 and ord(c) <= 0x206f  ) or ord(c) == 0xfeff
 
      def is_special_char(c):
        return ( ord(c) >= 0x0370 )

      temp_token = "d"
      for c in token:
        if not is_special_char(c):
          temp_token += c
      
      if temp_token == "":
        print ('SPECIAL TOKEN', temp_token , " , " , token)

      e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "skip_token": ("<" in token) or ( temp_token == "" ) })
  #print ('create_example_from_jsonl\'s e: ' , e.keys() )
  add_candidate_types_and_positions(e)
  #print ('after add_candidate_types_and_positions\'s e: document_tokens' , e['document_tokens'][:20] , 'long_answer_candidates' , e['long_answer_candidates'][:10])
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)
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
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])
    
  #print( 'question : ' , question )
  #print( 'answer : ' , answer )

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
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

  single_map = []
  single_context = []
  offset = 0
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("%d %s" %
                          (context["id"], context["type"]))
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
        tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
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

# We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
_DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length","include_trainset"])

def _check_is_max_context(doc_spans, cur_span_index, position):
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
  features = list()  

  query_tokens = tokenization.encode_ids(
      tokenizer.sp_model,
      tokenization.preprocess_text(
          example.questions[-1], lower=FLAGS.do_lower_case))

  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[0:FLAGS.max_query_length]

  paragraph_text = ' '.join(example.doc_tokens)
  parag_text_split = example.doc_tokens
  len_of_parag_text_split = len(parag_text_split)
  para_tokens = tokenization.encode_pieces(
      tokenizer.sp_model,
      tokenization.preprocess_text(paragraph_text, lower=FLAGS.do_lower_case),
      return_unicode=False)

  para_tokens_ = []
  for para_token in para_tokens:
    if type(para_token) == bytes:
      para_token = para_token.decode("utf-8")
    para_tokens_.append(para_token)
  para_tokens = para_tokens_
      
  tok_start_to_orig_index = []
  tok_end_to_orig_index = []
  parag_idx = 0
  token_to_parag_index = []
  parag_to_token_index = []
  token_to_origin_index = []
  temp_doc_parag_token = [ '"' if p == '``' or p == '\'\'' else p.lower() for p in parag_text_split ]
  parag_to_token_index = [ [100000 , -1 ] for _t in parag_text_split ]
  overflow_offset_cnt = 5
  for i,t_token in enumerate(para_tokens):
    #print('A' , t_token) 
    str_tok = t_token.replace(tokenization.SPIECE_UNDERLINE.decode("utf-8"), "")
    #print('B' , str_tok)
    #print('BB' , parag_text_split[parag_idx] )
    print('%s , %s' % (str_tok ,  temp_doc_parag_token[parag_idx]) )
    
    match_parag_idx = -1
    cur_parag_idx = parag_idx
    offset_idx = 0
    
    while True:
      prev_para_token = None
      after_para_token = None
      match_para_token = None
      if parag_idx + offset_idx < len(parag_text_split):
        #after_para_token = unicodedata.normalize('NFKD', temp_doc_parag_token[parag_idx + offset_idx]).encode('ascii', 'ignore').decode('utf8') 
        after_para_token = tokenization.preprocess_text(temp_doc_parag_token[parag_idx + offset_idx],FLAGS.do_lower_case)
        #after_para_token = temp_doc_parag_token[parag_idx + offset_idx ]
        #print ( 'offset_idx : %s %s %s'% ( parag_idx + offset_idx ,  temp_doc_parag_token[parag_idx + offset_idx ] ,  after_para_token  ) )
      if parag_idx - offset_idx >= 0 and parag_idx - offset_idx < len(parag_text_split)  and offset_idx < 3:
        #prev_para_token = temp_doc_parag_token[parag_idx - offset_idx ]
        prev_para_token = tokenization.preprocess_text(temp_doc_parag_token[parag_idx - offset_idx],FLAGS.do_lower_case)
        #print ( 'offset_idx : %s %s %s' % (parag_idx - offset_idx ,  temp_doc_parag_token[parag_idx - offset_idx ] , prev_para_token  ) )
      if after_para_token != None and ( str_tok == temp_doc_parag_token[parag_idx + offset_idx ] or (str_tok == after_para_token ) or  (str_tok in temp_doc_parag_token[parag_idx + offset_idx ]) or (str_tok in after_para_token ) ) :
      #if after_para_token != None and ( (str_tok == after_para_token ) or (str_tok in after_para_token ) ) :
        cur_parag_idx = parag_idx + offset_idx
        overflow_offset_cnt = 5
        match_para_token = after_para_token  
        break
      elif prev_para_token != None and parag_idx + offset_idx != parag_idx - offset_idx and (   str_tok == temp_doc_parag_token[parag_idx - offset_idx ] or  str_tok==prev_para_token or    ( str_tok in temp_doc_parag_token[parag_idx - offset_idx ]) or ( str_tok in prev_para_token )  ):
      #elif prev_para_token != None and parag_idx + offset_idx != parag_idx - offset_idx and ( ( str_tok==prev_para_token) or ( str_tok in prev_para_token )  ):
        cur_parag_idx = parag_idx - offset_idx
        overflow_offset_cnt = 5
        match_para_token = prev_para_token
        break
      offset_idx += 1
      #assert offset_idx < 5
      if offset_idx > overflow_offset_cnt:
        #print( 'offset_idx OVERFLOW!!! %s'%overflow_offset_cnt)
        overflow_offset_cnt=overflow_offset_cnt+1
        #if parag_idx + 1 < len(parag_text_split):
        #  cur_parag_idx = parag_idx + 1
        break
      if overflow_offset_cnt > 20:
        #print( ' overflow_offset_cnt > 30!!!')
        break
    if parag_idx != cur_parag_idx:
          parag_idx = cur_parag_idx
    parag_to_token_index[parag_idx][0] = i if i < parag_to_token_index[parag_idx][0] else parag_to_token_index[parag_idx][0]
    parag_to_token_index[parag_idx][1] = i if i > parag_to_token_index[parag_idx][1] else parag_to_token_index[parag_idx][1]

    token_to_parag_index.append(parag_idx)
    #print('C',temp_doc_parag_token[parag_idx])
    
    if str_tok == temp_doc_parag_token[cur_parag_idx] or str_tok == match_para_token:
      if parag_idx+1 < len(parag_text_split):
        parag_idx += 1
    elif len(str_tok) < len(temp_doc_parag_token[cur_parag_idx]) and len(str_tok) > 2:
      if str_tok == temp_doc_parag_token[cur_parag_idx][-len(str_tok):]:
        if parag_idx+1 < len(parag_text_split):
          parag_idx += 1

    '''
    for t_idx,ch in enumerate(str_tok):
  

      #print ( 'char a:', ch , 'char b:'  , temp_doc_parag_token[parag_idx][0] )
      #assert ch == temp_doc_parag_token[parag_idx][0]
      if len(temp_doc_parag_token[parag_idx]) > 1:
        temp_doc_parag_token[parag_idx] = temp_doc_parag_token[parag_idx][1:]
      else:
        temp_doc_parag_token[parag_idx] = ""
    '''
  #print(example.doc_tokens)
  print(len(example.doc_tokens) , len(example.doc_tokens_map) )
  print ( len(example.doc_tokens_map) , len(token_to_parag_index) )
  #print( token_to_parag_index )
  token_to_origin_index = [ example.doc_tokens_map[index] for index in token_to_parag_index ]

  #print ( len(token_to_parag_index), len(tok_start_to_orig_index) )
  assert len(token_to_parag_index) == len(para_tokens)
  if len_of_parag_text_split != len(parag_to_token_index):
    print(example.doc_tokens)
    print(len_of_parag_text_split , len(parag_to_token_index))
  assert len_of_parag_text_split == len(parag_to_token_index)
  
  tok_start_position = 0
  tok_end_position = 0

  if is_training and ( example.start_position != None and example.end_position != None ):
    tok_start_position = parag_to_token_index[example.start_position][0]

    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = parag_to_token_index[example.end_position][1]
    else:
      tok_end_position = parag_to_token_index[len(example.doc_tokens) - 1][1]
    
    if tok_start_position > tok_end_position:
      tok_end_position = tok_start_position
    elif tok_end_position < 0 or tok_start_position < 0:
        tok_start_position = 0
        tok_end_position = 0

    assert tok_start_position <= tok_end_position

  def _piece_to_id(x):
    if six.PY2 and isinstance(x, six.text_type):
      x = six.ensure_binary(x, "utf-8")
    return tokenizer.sp_model.PieceToId(x)

  all_doc_tokens = list(map(_piece_to_id, para_tokens))

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    doc_spans.append(_DocSpan(start=start_offset, length=length , include_trainset=False))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  doc_shuffle_idx = list(range(len(doc_spans)))
  random.shuffle(doc_shuffle_idx) 
  for t_idx in range(2):
    if t_idx >= len(doc_shuffle_idx):
        continue
    s_idx = doc_shuffle_idx[t_idx]
    doc_spans[s_idx] = _DocSpan(start=doc_spans[s_idx].start, length=doc_spans[s_idx].length , include_trainset=True)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_is_max_context = {}
    segment_ids = []
    p_mask = []
    cls_index = 0

    cur_token_to_origin_index = []

    tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
    segment_ids.append(0)
    p_mask.append(0)
    cur_token_to_origin_index.append(-1)
    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
      p_mask.append(1)
      cur_token_to_origin_index.append(-1)
    tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
    segment_ids.append(0)
    p_mask.append(1)
    cur_token_to_origin_index.append(-1)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i

      is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
      p_mask.append(0)
      cur_token_to_origin_index.append(int(token_to_origin_index[split_token_index]))
    tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
    segment_ids.append(1)
    p_mask.append(1)
    cur_token_to_origin_index.append(-1)

    assert len(cur_token_to_origin_index) == len(segment_ids)
  
    paragraph_len = len(tokens)
    input_ids = tokens

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < FLAGS.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      p_mask.append(1)
      cur_token_to_origin_index.append(-1)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length
    assert len(cur_token_to_origin_index) == FLAGS.max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)
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
        print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(token_to_origin_index[doc_start] , token_to_origin_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%(token_to_origin_index[tok_start_position] , token_to_origin_index[tok_end_position] ))
        print ('ANSWER_ERROR : %s target_answer_type != answer_type : %s != %s'%(example.example_id , target_answer_type , answer_type))
        #print ('ANSWER_ERROR : %s target_answer_type != answer_type : %s BBBBB %s'%(example.example_id , tokens , token_to_orig_map ) )
      elif answer_type != AnswerType.UNKNOWN:
        if target_answer_type == 3:
            has_answer = False
            for positions_tup in target_short_answers:
                if cur_token_to_origin_index[start_position] == positions_tup[0] and cur_token_to_origin_index[end_position] == positions_tup[1]-1:
                    has_answer = True
            if not has_answer:
                #print ( target_dict )
                print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(token_to_origin_index[doc_start] , token_to_origin_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%( token_to_origin_index[tok_start_position] , token_to_origin_index[tok_end_position] ))
                print ('ANSWER_ERROR: %s %s type : %s != %s'%(example.example_id , AnswerType.get(target_answer_type) , (cur_token_to_origin_index[start_position] , cur_token_to_origin_index[end_position]) , target_short_answers ))
                #print ('ANSWER_ERROR: %s %s type : %s BBBBB %s'%(example.example_id , AnswerType.get(target_answer_type) , tokens , token_to_orig_map ) )
        elif target_answer_type == 4:
            has_answer = False
            for positions_tup in target_long_answer:
                if cur_token_to_origin_index[start_position] == positions_tup[0] +1  and cur_token_to_origin_index[end_position] == positions_tup[1]-2:
                    has_answer = True
            if not has_answer:
                #print ( target_dict )
                print (example.example_id , 'DOC_SPAN_START : ' , doc_start , 'DOC_SPAN_END : ' , doc_end , 'REAL POSITION : (%s, %s)'%(  token_to_origin_index[doc_start] , token_to_origin_index[doc_end] ) , 'TARGET POSITION : (%s, %s)'%( token_to_origin_index[tok_start_position] , token_to_origin_index[tok_end_position] ))
                print ('ANSWER_ERROR: %s %s type : %s != %s'%(example.example_id , AnswerType.get(target_answer_type) , (cur_token_to_origin_index[start_position] , cur_token_to_origin_index[end_position]) , target_long_answer ))
                #print ('ANSWER_ERROR: %s %s type : %s BBBBB %s'%(example.example_id , AnswerType.get(target_answer_type) , tokens , token_to_orig_map ) )
      
      answer_text = ' '.join(parag_text_split[cur_token_to_origin_index[start_position]:cur_token_to_origin_index[end_position]+1])
    print ('making for %s , %s ' % ( example.example_id , doc_span ))
    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=cur_token_to_origin_index,
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
    
def nq_examples_iter(input_file, is_training):
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
      for index, line in enumerate(input_file):
        #print ( index , ' th line')
        #print ( 'line : ' , line)
        entry = create_example_from_jsonl(line)
        #print ( 'entry : '  , entry )
        yield read_nq_entry(entry, is_training)
    
def read_nq_examples(input_file, is_training,tqdm=None):
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
      for index, line in enumerate(input_file):
        input_data.append(create_example_from_jsonl(line))
        # if index > 100:
        #     break

  examples = []
  for entry in input_data:
    examples.extend(read_nq_entry(entry, is_training))
  return examples

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
    features["token_map"] = create_int_feature(feature.token_to_orig_map)

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



def top_k_indices(logits,n_best_size,token_map):
    #print ('sort :' , np.sort(logits[1:]))
    indices = np.argsort(logits[1:])+1
    indices = indices[token_map[indices]!=-1]
    return indices[-n_best_size:]
    
'''    
def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = FLAGS.n_best_size
  max_answer_length = FLAGS.max_answer_length
  i = 0
  score_map = dict()
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
    #print ( 'start_indexes[None] : ' , start_indexes[None])
    #print ( 'end_indexes[:,None] : ' , end_indexes[:,None])
    indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
    #print ( indexes )

    #print ( '(indexes[:,0]<indexes[:,1]) : ' , (indexes[:,0]<indexes[:,1]) )
    #print ( '(indexes[:,1]-indexes[:,0]<max_answer_length) : ' , (indexes[:,1]-indexes[:,0]<max_answer_length) )
    indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]

    cls_token_start_score = result.start_logits[0]
    cls_token_end_score = result.end_logits[0]
    for start_index,end_index in indexes:
        start_score = result.start_logits[start_index]
        end_score = result.end_logits[end_index]
        summary = ScoreSummary()
        summary.short_span_score = ( start_score + end_score)/2.0
        summary.cls_token_score = ( cls_token_start_score + cls_token_end_score)/2.0
        summary.answer_type_logits = result.answer_type_logits
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1
        
        if start_span not in score_map:
              score_map[start_span] = list()
        score_map[start_span].append(start_score)

        if end_span not in score_map:
              score_map[end_span] = list()
        score_map[end_span].append(end_score)

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, i, summary, start_span, end_span))
        i += 1 # to break ties

  # Default empty prediction.
  score = -10000.0
  short_span = Span(-1, -1)
  long_span  = Span(-1, -1)
  summary    = ScoreSummary()

  if predictions:
    score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": int(example.example_id),
      "long_answer": {
          "start_token": int(long_span.start_token_idx),
          "end_token": int(long_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": float(score),
      "short_answers": [{
          "start_token": int(short_span.start_token_idx),
          "end_token": int(short_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answer_score": float(score),
      "yes_no_answer": "NONE",
      "answer_type_logits": summary.answer_type_logits.tolist(),
      "answer_type": int(np.argmax(summary.answer_type_logits))
  }

  return summary
'''

def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = FLAGS.n_best_size
  max_answer_length = FLAGS.max_answer_length
  i = 0
  score_map = dict()
  cls_token_score_list = list()
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
    #print ( 'start_indexes[None] : ' , start_indexes[None])
    #print ( 'end_indexes[:,None] : ' , end_indexes[:,None])
    indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
    #print ( indexes )

    #print ( '(indexes[:,0]<indexes[:,1]) : ' , (indexes[:,0]<indexes[:,1]) )
    #print ( '(indexes[:,1]-indexes[:,0]<max_answer_length) : ' , (indexes[:,1]-indexes[:,0]<max_answer_length) )
    indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]

    cls_token_start_score = result.start_logits[0]
    cls_token_end_score = result.end_logits[0]
    cls_token_score = ( cls_token_start_score + cls_token_end_score)/2.0
    cls_token_score_list.append(cls_token_score)
    for start_index,end_index in indexes:
        #start_scores = list()
        #end_scores = list()
        #if start_index > 0 and token_map[start_index-1] != -1:
        #    start_scores.append(result.start_logits[start_index-1])
        #if end_index > 0 and token_map[end_index-1] != -1:
        #    end_scores.append(result.end_logits[end_index-1])
        
        #if start_index < FLAGS.max_seq_length-2 and token_map[start_index+1] != -1:  
        #    start_scores.append(result.start_logits[start_index+1])
        #if end_index < FLAGS.max_seq_length-2 and token_map[end_index+1] != -1:  
        #    end_scores.append(result.end_logits[end_index+1])   
    
        #start_scores.append(result.start_logits[start_index])
        #end_scores.append(result.end_logits[end_index])
        #print("LEN of start_scores : " , len( start_scores ))        
        #print("LEN of end_scores : " , len( end_scores )) 
        #start_score = sum(start_scores) / len(start_scores)
        #end_score = sum(end_scores) / len(end_scores)
        start_score = result.start_logits[start_index]
        end_score = result.end_logits[end_index]
        summary = ScoreSummary()
        summary.short_span_score = ( start_score + end_score)/2.0
        summary.cls_token_score = cls_token_score
        summary.answer_type_logits = result.answer_type_logits
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1
        '''
        if start_span not in score_map:
              score_map[start_span] = list()
        score_map[start_span].append(start_score)

        if end_span not in score_map:
              score_map[end_span] = list()
        score_map[end_span].append(end_score)
        '''

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, i, summary, start_span, end_span , summary.cls_token_score))
        i += 1 # to break ties
  #cls_token_score = sum(cls_token_score_list) / len(cls_token_score_list)
  # Default empty prediction.
  score = -10000.0
  short_span = Span(-1, -1)
  long_span  = Span(-1, -1)
  cls_score = 0.
  summary    = ScoreSummary()

  if predictions:
    score, _, summary, start_span, end_span,cls_score = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": int(example.example_id),
      "long_answer": {
          "start_token": int(long_span.start_token_idx),
          "end_token": int(long_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": float(score),
      "short_answers": [{
          "start_token": int(short_span.start_token_idx),
          "end_token": int(short_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      }],
      "cls_score": cls_score,
      "short_answer_score": float(score),
      "yes_no_answer": "NONE",
      "answer_type_logits": summary.answer_type_logits.tolist(),
      "answer_type": int(np.argmax(summary.answer_type_logits))
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
'''
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
    for start_index,end_index in indexes:
        summary = ScoreSummary()
        summary.short_span_score = (
            result.start_logits[start_index] +
            result.end_logits[end_index])
        summary.cls_token_score = (
            result.start_logits[0] + result.end_logits[0])
        summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, i, summary, start_span, end_span))
        i += 1 # to break ties

  # Default empty prediction.
  score = -10000.0
  short_span = Span(-1, -1)
  long_span  = Span(-1, -1)
  summary    = ScoreSummary()

  if predictions:
    score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": int(example.example_id),
      "long_answer": {
          "start_token": int(long_span.start_token_idx),
          "end_token": int(long_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": float(score),
      "short_answers": [{
          "start_token": int(short_span.start_token_idx),
          "end_token": int(short_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answer_score": float(score),
      "yes_no_answer": "NONE",
      "answer_type_logits": summary.answer_type_logits.tolist(),
      "answer_type": int(np.argmax(summary.answer_type_logits))
  }

  return summary
'''

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
                 doc_stride=128,
                 include_unknowns=1.0,
                 n_best_size=5,
                 max_answer_length=30)

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
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][0]["end_token"]
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
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    tf.compat.v1.logging.warning("Unknown candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(e,no_skip_idx):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e,no_skip_idx):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
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
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][0]["end_token"],
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
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
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
  features = list()  

  query_tokens = tokenization.encode_ids(
      tokenizer.sp_model,
      tokenization.preprocess_text(
          example.questions[-1], lower=FLAGS.do_lower_case))

  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[0:FLAGS.max_query_length]

  paragraph_text = ' '.join(example.doc_tokens)
  paragraph_text1 = tokenization.preprocess_text(paragraph_text, lower=FLAGS.do_lower_case)
  parag_text_split = example.doc_tokens
  parag_text_split1 = paragraph_text1.split()
  len_of_parag_text_split = len(parag_text_split)
  assert len(parag_text_split1) == len_of_parag_text_split
  para_tokens = tokenization.encode_pieces(
      tokenizer.sp_model,
      tokenization.preprocess_text(paragraph_text, lower=FLAGS.do_lower_case),
      return_unicode=False)

  para_tokens_ = []
  for para_token in para_tokens:
    if type(para_token) == bytes:
      para_token = para_token.decode("utf-8")
    para_tokens_.append(para_token)
  para_tokens = para_tokens_
      
  tok_start_to_orig_index = []
  tok_end_to_orig_index = []
  parag_idx = 0
  token_to_parag_index = []
  parag_to_token_index = []
  token_to_origin_index = []
  temp_doc_parag_token = [ '"' if p == '``' or p == '\'\'' else p.lower() for p in parag_text_split ]
  parag_to_token_index = [ [100000 , -1 ] for _t in parag_text_split ]
  overflow_offset_cnt = 5
  for i,t_token in enumerate(para_tokens):
    #print('A' , t_token) 
    str_tok = t_token.replace(tokenization.SPIECE_UNDERLINE.decode("utf-8"), "")
    #print('B' , str_tok)
    #print('BB' , parag_text_split[parag_idx] )
    print('%s , %s' % (str_tok ,  temp_doc_parag_token[parag_idx]) )
    
    match_parag_idx = -1
    cur_parag_idx = parag_idx
    offset_idx = 0
    
    while True:
      prev_para_token = None
      after_para_token = None
      match_para_token = None
      if parag_idx + offset_idx < len(parag_text_split):
        #after_para_token = unicodedata.normalize('NFKD', temp_doc_parag_token[parag_idx + offset_idx]).encode('ascii', 'ignore').decode('utf8') 
        after_para_token = tokenization.preprocess_text(temp_doc_parag_token[parag_idx + offset_idx],FLAGS.do_lower_case)
        #after_para_token = temp_doc_parag_token[parag_idx + offset_idx ]
        #print ( 'offset_idx : %s %s %s'% ( parag_idx + offset_idx ,  temp_doc_parag_token[parag_idx + offset_idx ] ,  after_para_token  ) )
      if parag_idx - offset_idx >= 0 and parag_idx - offset_idx < len(parag_text_split)  and offset_idx < 3:
        #prev_para_token = temp_doc_parag_token[parag_idx - offset_idx ]
        prev_para_token = tokenization.preprocess_text(temp_doc_parag_token[parag_idx - offset_idx],FLAGS.do_lower_case)
        #print ( 'offset_idx : %s %s %s' % (parag_idx - offset_idx ,  temp_doc_parag_token[parag_idx - offset_idx ] , prev_para_token  ) )
      if after_para_token != None and ( str_tok == temp_doc_parag_token[parag_idx + offset_idx ] or (str_tok == after_para_token ) or  (str_tok in temp_doc_parag_token[parag_idx + offset_idx ]) or (str_tok in after_para_token ) ) :
      #if after_para_token != None and ( (str_tok == after_para_token ) or (str_tok in after_para_token ) ) :
        cur_parag_idx = parag_idx + offset_idx
        overflow_offset_cnt = 5
        match_para_token = after_para_token  
        break
      elif prev_para_token != None and parag_idx + offset_idx != parag_idx - offset_idx and (   str_tok == temp_doc_parag_token[parag_idx - offset_idx ] or  str_tok==prev_para_token or    ( str_tok in temp_doc_parag_token[parag_idx - offset_idx ]) or ( str_tok in prev_para_token )  ):
      #elif prev_para_token != None and parag_idx + offset_idx != parag_idx - offset_idx and ( ( str_tok==prev_para_token) or ( str_tok in prev_para_token )  ):
        cur_parag_idx = parag_idx - offset_idx
        overflow_offset_cnt = 5
        match_para_token = prev_para_token
        break
      offset_idx += 1
      #assert offset_idx < 5
      if offset_idx > overflow_offset_cnt:
        #print( 'offset_idx OVERFLOW!!! %s'%overflow_offset_cnt)
        overflow_offset_cnt=overflow_offset_cnt+1
        #if parag_idx + 1 < len(parag_text_split):
        #  cur_parag_idx = parag_idx + 1
        break
      if overflow_offset_cnt > 20:
        #print( ' overflow_offset_cnt > 30!!!')
        break
    if parag_idx != cur_parag_idx:
          parag_idx = cur_parag_idx
    parag_to_token_index[parag_idx][0] = i if i < parag_to_token_index[parag_idx][0] else parag_to_token_index[parag_idx][0]
    parag_to_token_index[parag_idx][1] = i if i > parag_to_token_index[parag_idx][1] else parag_to_token_index[parag_idx][1]

    token_to_parag_index.append(parag_idx)
    #print('C',temp_doc_parag_token[parag_idx])
    
    if str_tok == temp_doc_parag_token[cur_parag_idx] or str_tok == match_para_token:
      if parag_idx+1 < len(parag_text_split):
        parag_idx += 1
    elif len(str_tok) < len(temp_doc_parag_token[cur_parag_idx]) and len(str_tok) > 2:
      if str_tok == temp_doc_parag_token[cur_parag_idx][-len(str_tok):]:
        if parag_idx+1 < len(parag_text_split):
          parag_idx += 1

    '''
    for t_idx,ch in enumerate(str_tok):
  

      #print ( 'char a:', ch , 'char b:'  , temp_doc_parag_token[parag_idx][0] )
      #assert ch == temp_doc_parag_token[parag_idx][0]
      if len(temp_doc_parag_token[parag_idx]) > 1:
        temp_doc_parag_token[parag_idx] = temp_doc_parag_token[parag_idx][1:]
      else:
        temp_doc_parag_token[parag_idx] = ""
    '''
  #print(example.doc_tokens)
  print(len(example.doc_tokens) , len(example.doc_tokens_map) )
  print ( len(example.doc_tokens_map) , len(token_to_parag_index) )
  #print( token_to_parag_index )
  token_to_origin_index = [ example.doc_tokens_map[index] for index in token_to_parag_index ]

  #print ( len(token_to_parag_index), len(tok_start_to_orig_index) )
  assert len(token_to_parag_index) == len(para_tokens)
  if len_of_parag_text_split != len(parag_to_token_index):
    print(example.doc_tokens)
    print(len_of_parag_text_split , len(parag_to_token_index))
  assert len_of_parag_text_split == len(parag_to_token_index)
  
  tok_start_position = 0
  tok_end_position = 0

  if is_training and ( example.start_position != None and example.end_position != None ):
    tok_start_position = parag_to_token_index[example.start_position][0]

    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = parag_to_token_index[example.end_position][1]
    else:
      tok_end_position = parag_to_token_index[len(example.doc_tokens) - 1][1]
    
    if tok_end_position < 0 or tok_start_position < 0:
        print('ERROR')
        tok_start_position = 0
        tok_end_position = 0

    assert tok_start_position <= tok_end_position

  def _piece_to_id(x):
    if six.PY2 and isinstance(x, six.text_type):
      x = six.ensure_binary(x, "utf-8")
    return tokenizer.sp_model.PieceToId(x)

  all_doc_tokens = list(map(_piece_to_id, para_tokens))
  assert len(all_doc_tokens) == len(para_tokens)
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
    answer_span_cnt = 1 #if len(doc_spans) == 0 else 0
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
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      length = min(length, max_tokens_for_doc)
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
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = token_to_origin_index[split_token_index]

      #is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
      token_is_max_context[len(tokens)] = False # is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)
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
      if (not contains_an_annotation ):
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
                for positions_tup in target_short_answers:
                    if token_to_orig_map[start_position] == positions_tup[0] and token_to_orig_map[end_position]+1 == positions_tup[1]:
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
        
      answer_text = " ".join(tokens[start_position:(end_position + 1)])
    print ('making for %s , %s ' % ( example.example_id , doc_span ))
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

    
'''    
def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = FLAGS.n_best_size
  max_answer_length = FLAGS.max_answer_length
  i = 0
  score_map = dict()
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
    #print ( 'start_indexes[None] : ' , start_indexes[None])
    #print ( 'end_indexes[:,None] : ' , end_indexes[:,None])
    indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
    #print ( indexes )

    #print ( '(indexes[:,0]<indexes[:,1]) : ' , (indexes[:,0]<indexes[:,1]) )
    #print ( '(indexes[:,1]-indexes[:,0]<max_answer_length) : ' , (indexes[:,1]-indexes[:,0]<max_answer_length) )
    indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]

    cls_token_start_score = result.start_logits[0]
    cls_token_end_score = result.end_logits[0]
    for start_index,end_index in indexes:
        start_score = result.start_logits[start_index]
        end_score = result.end_logits[end_index]
        summary = ScoreSummary()
        summary.short_span_score = ( start_score + end_score)/2.0
        summary.cls_token_score = ( cls_token_start_score + cls_token_end_score)/2.0
        summary.answer_type_logits = result.answer_type_logits
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1
        
        if start_span not in score_map:
              score_map[start_span] = list()
        score_map[start_span].append(start_score)

        if end_span not in score_map:
              score_map[end_span] = list()
        score_map[end_span].append(end_score)

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, i, summary, start_span, end_span))
        i += 1 # to break ties

  # Default empty prediction.
  score = -10000.0
  short_span = Span(-1, -1)
  long_span  = Span(-1, -1)
  summary    = ScoreSummary()

  if predictions:
    score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": int(example.example_id),
      "long_answer": {
          "start_token": int(long_span.start_token_idx),
          "end_token": int(long_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": float(score),
      "short_answers": [{
          "start_token": int(short_span.start_token_idx),
          "end_token": int(short_span.end_token_idx),
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answer_score": float(score),
      "yes_no_answer": "NONE",
      "answer_type_logits": summary.answer_type_logits.tolist(),
      "answer_type": int(np.argmax(summary.answer_type_logits))
  }

  return summary
'''

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
            summary.cls_token_score = (
                result.start_logits[0] + result.end_logits[0])
            summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1

            answer_type = int(np.argmax(summary.answer_type_logits))
            start_logit = int(np.argmax(result.start_logits))
            end_logit = int(np.argmax(result.end_logits))
            
            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            #if start_logit != 0 and end_logit != 0 and answer_type != 0:
            predictions.append( (score, i, summary, start_span, end_span ) )
            '''
            summary.short_span_score = (
                result.start_logits[start_index] +
                result.end_logits[end_index])/2.
            summary.cls_token_score = (
                result.start_logits[0] + result.end_logits[0])/2.
            summary.answer_type_logits = result.answer_type_logits #-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1
            answer_type = int(np.argmax(summary.answer_type_logits))
            start_logit = int(np.argmax(result.start_logits))
            end_logit = int(np.argmax(result.end_logits))
            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score # - summary.cls_token_score
            if start_logit != 0 and end_logit != 0 and answer_type != 0:
              predictions.append((score, i, summary, start_span, end_span,summary.answer_type_logits))
            '''
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



