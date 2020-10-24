import tkinter as tk
from tkinter import ttk as ttk
import numpy as np
from random import seed
from random import randint
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras.models import load_model

seed(1)

def text_generate(text):
    res = [['if you having a bad day , just remember that you have managed to get through every bad day you\'ve had. you\'ll make it through this one too. ', 
    'Bad days makes you stronger, those are the best days because God is talking to you. you just have to listen closely. Don\'t let your tears become louder than his voice.',
    'Trust me, You have to fight bad days to have best days in your Life. '], 
    ['The boring thing with taking a walk with someone is that your thoughts are then dictated by the subject or subjects of your conversation; and that is made worse by the fact that most sane people are terrified of silence whenever they are with or near someone. ', 
    'For most people, life would be boring without meaningful work.', 
    'Boring is a syndrome of being too simple.'], 
    ['Boring is a syndrome of being too simple.  ', 
    'I decided I would fill the emptiness in me with God and with paint.', 
    'Life is an empty bottle filled with love.'], 
    ['Live your truth. Express your love. Share your enthusiasm. Take action towards your dreams. Walk your talk. Dance and sing to your music. Embrace your blessings. Make today worth remembering.', 
    'It\'s faith in something and enthusiasm for something that makes a life worth living.', 
    'Not a visible enthusiasm but a hidden one, an excitement burning with a cold flame.'], 
    ['Fears are educated into us, and can, if we wish, be educated out.', 
    'Curiosity will conquer fear even more than bravery will.', 
    'I am not afraid of tomorrow, for I have seen yesterday and I love today.'], 
    ['The best thing about the future is that it comes one day at a time.', 
    'The difference between stupidity and genius is that genius has its limits.', 
    'Facebook just sounds like a drag, in my day seeing pictures of peoples vacations was considered a punishment.'], 
    ['Love is that condition in which the happiness of another person is essential to your own.', 
    'Happiness is when what you think, what you say, and what you do are in harmony.', 
    'You will never be happy if you continue to search for what happiness consists of. You will never live if you are looking for the meaning of life.'], 
    ['Hatters dont really hate you, they hate themselves because you are a reflection of what they wish to be', 
    'Dont hate what you don\'t understand', 
    'Enjoy who you are. dont hate yourself for what you aren\'t'], 
    ['I sustain myself with the love of family ', 
    'It is in the love of one\'s family only that heartfelt happiness is known', 
    'The love of a family is life\'s greatest blessing'], 
    ['People who demand neutrality in any situation are usually not neutral but in favor of status quo', 
    'Simple thing become complicated when you expect too much', 
    'The primary cause of unhappiness is never the situation but your thoughts about it.'], 
    ['God always has a relief for every sorrow and a plan for every tomorrow', 
    'An hour of anxiety cannot change my circumstances, but a minute of prayer can alter everything', 
    'Dont try to force anything let life be a deep let go. god opens millions of flowers everyday without forcing their buds'], 
    ['People cry not because they are weak, it\'s because they\'ve been strong for too long', 
    'Death is not the gratest loss in life. the greatest loss is what dies inside us while we live', 
    'The irony of grief is that the person that you need to talk to about how you feel is the person who is no longer here'], 
    ['Life is always full of surprises. you never know who you are going to meet that will change your life forever', 
    'Surprise is the gratest gift which life can grant us ', 
    'Life is full of surprises, some good, some not so good.'], 
    ['Worrying is a waste of time, good and bad things will happen in life. you just have to keep living and not stress over what you cant control', 
    'Stop worrying about someone that isn\'t worried about you', 
    'A day of worry is more exhausting than a day of work.']]
    list_of_classes = ['anger', 'boredom', 'empty', 'enthusiasm', 'fear', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']
    max_text_length = 400
    max_features = 20000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=max_features,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None,
        document_count=0,
    )
    model = load_model('emotion-classifier-keras-model.h5')
    tokenizer.fit_on_texts(list(text))
    x_tokenized = tokenizer.texts_to_sequences(text)
    text = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)  
    y_pred = model.predict(text)

    def pred(y_pred):
        y_pred = np.array(y_pred)
        list_of_classes = ['anger', 'boredom', 'empty', 'enthusiasm', 'fear', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']
        o = -1
        k = 0

        for i in range(len(list_of_classes)):
            if y_pred[i] > k:
                o = i
                k = y_pred[i]
    
        return o

    return res[pred(y_pred[0])][randint(0, 3)]

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.anss = tk.StringVar()
        self.anss.set('Text Generator')
        self.master = master
        self.pack()
        self.create_widgets()

    def gen(self):
        print('ok')
        preT = self.text.get()
        self.anss.set(text_generate(preT))
        return

    def create_widgets(self):
        self.topFrame = tk.Frame(self.master)
        self.topFrame['borderwidth'] = 0
        self.topFrame.place(height=20, width=40)
        self.topFrame.pack(side='top')

        textSentence = tk.Label(self.topFrame, text='Input Sentence')
        textSentence.pack(side='left')
        self.text = tk.Entry(self.topFrame, width = 100)
        self.text.place(height=100, width=100)
        self.text.pack(side='left')

        self.subButton = tk.Button(self.master, text='Generate', activeforeground='yellow', activebackground='blue', fg='green', command=self.gen)
        self.subButton.pack(side='top')

        self.bottomFrame = tk.Frame(self.master)
        self.bottomFrame['borderwidth'] = 0
        self.bottomFrame.pack(side='top')
        
        ansSentence = tk.Label(self.bottomFrame, text='Generated Text')
        ansSentence.pack(side='left')
        self.ans = ttk.Entry(self.bottomFrame, width = 100, textvariable=self.anss)
        self.ans.pack(side='left')


if __name__ == "__main__":
    root = tk.Tk()
    root.title('Text Generator GUI')
    app = Application(master = root)
    app.mainloop()


