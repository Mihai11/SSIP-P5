# https://www.programminginpython.com/create-temperature-converter-app-python-gui-using-tkinter/
# https://www.geeksforgeeks.org/python-gui-tkinter/
import tkinter as tk
from tkinter import ttk
import os
from tkinter import filedialog
from PIL import ImageTk, Image


class GUI(tk.Tk):

    def __init__(self, master=None):
        tk.Tk.__init__(self, master)
        self.title("SSIP 2019 -- Project No.5 -- Scan PDF --")
        self.geometry("800x500")
        self.configure(background='#19cefd')  # #00ff00
        self.grid()

        self.img = ImageTk.PhotoImage(Image.open("logo.jpg"))
        self.panel = tk.Label(master, image=self.img)
        self.panel.pack(side="bottom", fill="both", expand="yes")
        self.panel.grid(row=0, column=0)

        # Define variables:
        # ---------------------------------
        self.bytes = 0
        self.maxbytes = 0
        self.filename = tk.StringVar()
        self.dirname = tk.StringVar()
        self.mainPath = tk.StringVar()
        # ---------------------------------

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.lnameLabel = tk.Label(master, text="Upload one pdf file path: ", font=("bold"), height=2, bg='#19cefd')
        self.lnameLabel.grid(row=2, column=0)
        # --------
        self.fileNameEntry = tk.StringVar()
        self.fileNameEntry = tk.Entry(textvariable=self.fileNameEntry)
        self.fileNameEntry.grid(row=2, column=1)
        # --------
        self.submitButton = tk.Button(master, text="Submit", fg="red", font=("bold"), command=self.buttonClick)
        self.submitButton.grid(row=2, column=2)
        # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.lnameLabel = tk.Label(master, text="or", height=2, width=2, bg='#19cefd')
        self.lnameLabel.grid(row=3, column=0)
        # -------------------------------------------------------------------
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.lnameLabel = tk.Label(master, text="Search for PDF file path", font=("bold"), height=2, bg='#19cefd')
        self.lnameLabel.grid(row=4, column=0)
        self.fileExplorer = tk.Button(master, text="Search..", fg="red", font=("bold"), command=self.openFileLocation)
        self.fileExplorer.grid(row=4, column=1)
        # --------------------------------------------------------------------
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.lnameLabel = tk.Label(master, text="or", height=2, width=2, bg='#19cefd')
        self.lnameLabel.grid(row=5, column=0)
        # -------------------------------------------------------------------
        # --------------------------------------------------------------------
        # self.b = tk.Label(master, text="", bg='#19cefd')
        # self.b.grid(row=6, column=1)

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.lnameLabel = tk.Label(master, text="Search for directory path", font=("bold"), height=2, bg='#19cefd')
        self.lnameLabel.grid(row=7, column=0)
        self.fileExplorer = tk.Button(master, text="Search..", fg="green", font=("bold"), command=self.openFilesDirectory)
        self.fileExplorer.grid(row=7, column=1)
        # -------------------------------------------------------------------
        # --------------------------------------------------------------------

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        self.convertTriggerButton = tk.Button(master, text="Convert:", fg="green", font=("bold"),
                                              command=self.doSomething)
        self.convertTriggerButton.grid(row=8, column=4)

        # -------------------------------------------------------------------
        # --------------------------------------------------------------------

    def getFilePath(self):
        return self.mainPath

    def buttonClick(self):
        print("PDF path: " + self.fileNameEntry.get())
        self.mainPath = self.fileNameEntry.get()
        self.drawConfirmation(os.path.isfile(self.mainPath) or os.path.isdir(self.mainPath))

    def openFileLocation(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("PDF files", "*.pdf"), ("all files", "*.*")))
        self.mainPath = self.filename
        self.drawConfirmation(os.path.isfile(self.mainPath))

    def openFilesDirectory(self):
        self.dirname = filedialog.askdirectory(initialdir="/", title="Select directory")
        self.mainPath = self.dirname
        self.drawConfirmation(os.path.isdir(self.mainPath))

    def drawConfirmation(self, fileNameBoolean):
        if fileNameBoolean:
            self.lnameLabel.grid_forget()
            self.lnameLabel = tk.Label(self, text=self.mainPath, bg='#19cefd')
            self.lnameLabel.grid(row=8, column=5)
        else:
            self.lnameLabel = tk.Label(self, text="Invalid path!", bg='#19cefd')
            self.lnameLabel.grid(row=8, column=5)

    # call your functions below:
    # ------------------------------------------------
    def doSomething(self):

        print(self.mainPath)


if __name__ == "__main__":
    guiFrame = GUI()
    guiFrame.mainloop()
