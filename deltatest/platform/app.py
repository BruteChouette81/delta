#import keras

from tkinter import *
window=Tk()
# add widgets here

### load model:
selected_model = StringVar()

types = (("model - 3", "model3_20"),("model - 4", "model4_15"))
for type in types:
    r = Radiobutton(
        window,
        text=type[0],
        value=type[1],
        variable=selected_model
    )
    r.pack(fill='x', padx=10, pady=10)

button = Button(
    window,
    text="Load Model" ) #command=show_selected_size

button.pack(fill='x', padx=5, pady=5)

### input text
input = ""
lbl=Label(window, text="text Input:", fg='blue', font=("Helvetica", 12))
lbl.pack(fill="x",padx=20)
txtfld=Entry(window, bd=1)
txtfld.pack(fill="x", padx=12)

button2 = Button(
    window,
    text="submit text" ) #command=set input=txtfld.get(), model(input), txtfld2.insert(END, str(output))

button2.pack(fill='x', padx=5, pady=5)

### text output
output = ""
lbl2=Label(window, text="text output: ", fg='red', font=("Helvetica", 12))
lbl2.pack(fill="x",pady=20)
txtfld2=Entry(window, bd=1)
txtfld2.pack(fill="x", padx=12)

window.title('Delta Beta 0.5')
window.geometry("500x420")
window.mainloop()