from gc import callbacks
from tkinter import *

from gpt_test import train
callback = []
def start_train():
    print("start training")
    training.place(x=80, y=100)
    model, callbacks = train(4)
    callback.append(callbacks)
    model.save("model_4")
    training.config(text = "finish training")

    #epoch_label.config(text = epoch)

def get_callback():
    epoch_label.config(text="")
    epoch=int(call_back_entry.get())
    text = str(callback[0][epoch])
    epoch_label.config(text=text)

window=Tk()
# add widgets here

window.title('Delta (mark 4)')
# start training
btn=Button(window, text="Start training", fg='blue', bg='grey', font=("Helvetica", 16), command=start_train) # command=train()
btn.place(x=80, y=50)

training = Label(window, text="training...", fg='black', font=("Helvetica", 12))

call_label=Label(window, text="enter the epoch to get callback:", fg='black', font=("Helvetica", 12))
call_label.place(x=80, y=200)

call_back_entry = Entry(window, bd=3, )
call_back_entry.place(x=80, y=220, width=400, height=30)
sub_btn = Button(window, text="submit", fg='blue', bg='grey', font=("Helvetica", 14), command=get_callback) # command=train()
sub_btn.place(x=520, y=220)

epoch_label = Label(window, text="", fg='black', font=("Helvetica", 12))
epoch_label.place(x=80, y=300)


window.geometry("800x600+10+20")
window.mainloop()