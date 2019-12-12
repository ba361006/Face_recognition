import tkinter as tk
import Main as pj 
import os 
import shutil
from tkinter.font import Font
from tkinter import ttk


def Ok():
    if name_entry.get() == '':
        print('Error, please enter your name')
    else:
        pj.ok(name_entry.get())

def Capture_training_data():
    if name_entry.get() == '':
        print('Error, please enter your name')
    else:
        pj.capture_training_data(user_name = name_entry.get())

def Train():
    pj.train()

def Capture_testing_data():
    pj.capture_testing_data()
 
def Test():
    pj.eval()

def Upgrade():
    folders = []
    for name in os.listdir(pj.train_data_path):
        folders.append(name)
    combo_dropdown.config(values = folders)


def Delete():
    folder_name = combo_dropdown.get()
    if folder_name == 'Others':
        print('Error')
    else:
        shutil.rmtree(pj.train_data_path + folder_name)


window = tk.Tk()  
window.title('Project')
window.geometry('400x400')  
window.config(bg = '#323232')

ft_title = Font(family='Times new Roman', size=24, weight = 'bold')
ft_article = Font(family='Times new Roman', size=12)

header_label = tk.Label(window, bg='#323232', fg='white',font = ft_title, text='Face recognition')
header_label.pack(side = tk.TOP, pady = 20)


# name
name_frame = tk.Frame(window, bg='#323232')
name_frame.pack(side = tk.TOP)

name_words = tk.Label(name_frame, bg='#323232', fg='white', font = ft_article, text='1. Please enter your English name: ')
name_words.pack(side = tk.LEFT)

name_entry = tk.Entry(name_frame)
name_entry.pack(side = tk.LEFT)

name_ok = tk.Button(name_frame, font = ft_article, command = Ok, text = 'Ok')
name_ok.pack(side = tk.LEFT, padx = 5)


# train
train_frame = tk.Frame(window, bg='#323232')
train_frame.pack(side = tk.TOP, pady = 20)

train_words = tk.Label(train_frame, bg='#323232', fg='white', font = ft_article, text = '2. Train:                        ')
train_words.pack(side = tk.LEFT)

train_data = tk.Button(train_frame, font = ft_article, command = Capture_training_data, text = 'Capture training data')
train_data.pack(side = tk.LEFT, padx = 15)

train = tk.Button(train_frame, font = ft_article, command = Train, text = 'Train')
train.pack(side = tk.LEFT, padx = 15)


# test
test_frame = tk.Frame(window, bg = '#323232')
test_frame.pack(side = tk.TOP)

test_words = tk.Label(test_frame, bg='#323232', fg='white', font = ft_article, text = '3. Test:                         ')
test_words.pack(side = tk.LEFT)

test_data = tk.Button(test_frame, font = ft_article, command = Capture_testing_data, text = 'Capture testing data')
test_data.pack(side = tk.LEFT, padx = 20)

test = tk.Button(test_frame, font = ft_article, command = Test, text = 'Test')
test.pack(side = tk.LEFT, padx = 15)


# Delete
folders = []
for name in os.listdir(pj.train_data_path):
        folders.append(name)

combo_frame = tk.Frame(window, bg = '#323232')
combo_frame.pack(side = tk.TOP, pady = 20)

combo_words = tk.Label(combo_frame, bg = '#323232', fg = 'white', font = ft_article, text = '4. Delete:')
combo_words.pack(side = tk.LEFT)

combo_dropdown = ttk.Combobox(combo_frame, values = folders)
combo_dropdown.pack(side = tk.LEFT, padx = 10)

combo_upgrade = tk.Button(combo_frame, font = ft_article, command = Upgrade, text = 'Upgrade')
combo_upgrade.pack(side = tk.LEFT)

combo_delete = tk.Button(combo_frame, font = ft_article, command = Delete, text = 'Delete')
combo_delete.pack(side = tk.LEFT, padx = 17)


window.mainloop()
