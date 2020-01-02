
import json
import numpy as np
from utils import *
import random

#use to generate hints
def get_missing_key_words(list1, list2):
    return set(list2) - set(list2)


class AnswerRanker:
    texts = []
    def __init__(self, fname):
        self.fname = fname
        self.texts = self.get_corpus(fname)
        self.prev_answer = ""
        self.answered = []
        self.questions = [*self.texts]
        self.correct_answers = 0


    def get_corpus(self, fname):
        with open(fname) as f_in:
            return json.load(f_in)


    def ask_question(self):
        if len(self.questions) > 0 :
            return self.questions.pop(), False
        return "That's all the questions I have for now", True

    def evaluate_answer(self, answer):
        similarities = get_most_similar(answer, self.texts)
        certainty = np.max(similarities)
        if certainty < .2 :
            return 'give_hint'
        self.correct_answers += 1
        return 'congratulate'

    def generate_hint(self, answer, paragraph):
        # extract keywords and return a keyword that is in paragrapha nd not in answer then format as hmm, no tell me about %keyword or can have hints as part of json file
        return "Not, quite"


    def get_most_similar(self, keyword):
        texts = [text.split() for text in self.texts]
        print(texts)
        dictionary = corpora.Dictionary(texts)
        feature_cnt = len(dictionary.token2id)
        print(dictionary)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        print(tfidf)
        kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
        sim = index[tfidf[kw_vector]]
        most_similar = np.argmax(sim)
        """if sim > .3:
            text_keywords = get_keywords(texts[int(most_similar]))
            missing_keywords = get_missing_key_words(keyword, text_keywords)
            return missing_keywords
        else:
            return -1 """
        return -1


class DialogueManager:
    def __init__(self):
        # Intent recognition:
        self.intent_recognizer = IntentExtractor()

        self.HINT_TEMPLATE = 'You\'re close but have you thought about  %s?'
        self.QUESTION_TEMPLATE = 'Tell me about %s'

        # Goal-oriented part:
        self.answer_ranker = AnswerRanker('minicorpus.json')
        #bot will evaluate next answer
        self.waiting_for_answer = False
        #add missing keywods from hint generaot, when person hits all keywords mark question correct
        self.missing_keywords = []
        self.tries = 0


        # Chit-chat part
      #  self.create_chitchat_bot()


    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""
        self.chatbot = ChatBot('myBot')
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        #experimental random and weird corpus
        trainer.train('chatterbot.corpus.english')

    def generate_answer(self, user_input):
        """ self.categories = ["sayhi", "say_good", "say_not_much", "say_youre_welcome", "confirmP", "confirmN",
                           "give_hint", "prompt", "evaluate_answer", "answer_question", "sooth"]"""
        if self.waiting_for_answer and self.tries <=  1 :
            self.tries += 1
            intent = 'evaluate_answer'

        else:
            intent = self.intent_recognizer.find_intet(user_input)
        if intent == 'evaluate_answer':
            intent = self.answer_ranker.evaluate_answer(user_input)
        # intent may now be give hint or congratulate

        """if intent == "give_hint":
            self.waiting_for_answer = True
            return self.answer_ranker.generate_hint()"""

        if intent == 'dialogue':
            self.waiting_for_answer = False
            response = self.chatbot.get_response(user_input)
            return response
        if intent == 'prompt':
            self.tries = 0
            self.waiting_for_answer = True
            return self.answer_ranker.ask_question()
        else:
            self.waiting_for_answer = False
            return self.respond(intent)


    def respond(self,intent):

        # can generate these with neighbors of vectors
        responses = {"sayhi":['Hello!', 'Hi!', 'Hey!' ],
                     "say_good":['Great, what about you?', 'Just fine thank you'],
                     "say_not_much":['Not much.', 'Just here.'],
                     "say_youre_welcome":["You're welcome!"],
                    "confirmP":['Great!', "Alright", "OK"],
                    "confirmN":['Sorry to hear that.', "That sucks", "Wish I could help."],
                    "give_hint": ['Not quite, try again', "Give it another shot."],
                    "answer_question":['I\'m sorry', 'I cant answer that right now.' ],
                    "sooth":['I\m sorry to hear that', "I wish I could be more help," "I'm sorry :("],
                     "congratulate":["Great Job!", "Amazing", "You're doing great."]
        }
        return random.choice(responses[intent])


class IntentExtractor():
    # intent corpus should be expanded, real training corpus specifically within the domain of breast cancer is needed ^^)
    intents = {
        "greeting": ["hi", "hello", "hey", "good evening", "greetings", "greeting", "hi", "howdy"],
        "geetingQ": ["how are you", "how're you", "how are you doing",
                     "how ya doin'", "how ya doin", "how is everything", "how is everything going",
                     "how's everything going"],
        "greetingU": ["what is up", "what's up", "what is cracking", "what's cracking", "what is good", "what's good",
                      "what is happening", "what's happening", "what is new"],
        "thank_you": ["you're welcome!"],
        "confirmP": ["good", "fine", "ok", "pretty good", "cant complain", "sure"],
        "confirmN": ["bad", "not good", "pretty shit", "meh"],
        "need_hint": ["im stuck", "help", "idk", "i dont know", "wish i knew", "im not sure", "i need a hing",
                      "help me out", "i dont understand "],
        "move_on": ["yes","lets move on", "i dont want to talk about this anymore", "can we move on", "lets move",
                    "im not going to understand", "ask me another question", "im ready to learn", "ask me somehting"],
        "end": ["bye", "im leaving", "lets move on"],
        "answer": ["I think its something ", "i know that", "some treatments for cancer are",
                   "cancer is not a death sentence", "breast cancer can be overcome"],
        "question": ["whats going on with the cancer", "how can i know what to do", "who will help me",
                     "when am i going to feel better", "who can help me find treatment options"],
        "distress": ["i feel horrible", "im sick", "this is no help"],
        "dialogue": ["Random stuff goes here so it can go to chit chat "]
    }


    def __init__(self):
        self.self = self
        self.categories = ["sayhi", "say_good", "say_not_much", "say_youre_welcome", "confirmP", "confirmN",
                           "give_hint", "prompt", "evaluate_answer", "answer_question", "sooth", "dialogue"]

    def find_intet(self, text):
        most_common = get_most_similar(text, self.intents)
        intent = np.argmax(most_common)
        return self.categories[intent]


