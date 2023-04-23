import keras

from tkinter import *

from crystalAI import load_model, save_model, predict
from data.getData import load_crystal_vectorizer

window=Tk()
window.configure(bg='#171a25')
# add widgets here
def train(epochs: int):
    model = load_model("./deltatest/crystaltest/weights/crystalModel2")


    print(model.summary())
    model.compile("adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    dataset, vectorizer = load_crystal_vectorizer()
    model.fit(dataset, batch_size=64, epochs=epochs)
    save_model(model, "./deltatest/crystaltest/weights/crystalModel2") # accuracy: 0.6390, loss: 0.8378, No gradient desapearing


def interact():
    #set input=txtfld.get(), model(input), txtfld2.insert(END, str(output))
    prompt = txtfld.get()

    sentence = predict(prompt, [""])
    print(sentence)

    txtfld2.insert(END, str(sentence))

def show_selected_size():
    print(selected_model.get())

### load model:
selected_model = StringVar()

title = Label(window, text="Welcome back!", fg='white', bg='#171a25', font=("Helvetica", 28))
title.pack(side=TOP, anchor=NW, pady=50)

select = Label(window, text="Select your model:", fg='white', bg='#171a25', font=("Helvetica", 18))
select.pack(side=TOP, anchor=NW, padx=25, pady=30)

types = (("delta1", "crystalModel1"),("delta2", "crystalModel2"))
for type in types:
    r = Radiobutton(
        window,
        text=type[0],
        value=type[1],
        variable=selected_model,
        bg='#171a25',
        fg='white',
        font=("Helvetica", 16)
    )
    r.pack(side=TOP, anchor=NW, padx=25, pady=10)

button = Button(
    window,
    text="Load Model",
    bg='#131416',
    fg='#b2b7cc',
    font=("Helvetica", 12),
    command=show_selected_size ) #

button.pack(fill='x', padx=250, pady=25)

### input text
input = ""
lbl=Label(window, text="Chat with the selected model:", fg='white', bg='#171a25', font=("Helvetica", 16))
lbl.pack(side=TOP, anchor=NW, padx=25, pady=30)

txtfld=Entry(window, bd=1)
txtfld.pack(fill="x", padx=100)

button2 = Button(
    window,
    text="Enter",
    bg='#122a9f',
    fg='#b2b7cc',
    font=("Helvetica", 12),
    command=interact ) #command=set input=txtfld.get(), model(input), txtfld2.insert(END, str(output))

button2.pack(side=TOP, anchor=NW, padx=25, pady=30)

### text output
output = ""
lbl2=Label(window, text="Response: ", fg='white', bg='#171a25', font=("Helvetica", 16))
lbl2.pack(side=TOP, anchor=NW, padx=25, pady=30)
txtfld2=Entry(window, bd=1)
txtfld2.pack(fill="x", padx=100)

window.title('Crystal interactive library')
window.geometry("900x900")
window.mainloop()