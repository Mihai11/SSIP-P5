# https://www.programminginpython.com/create-temperature-converter-app-python-gui-using-tkinter/
# https://www.geeksforgeeks.org/python-gui-tkinter/
import tkinter as tk
from tkinter import ttk
from process_pdf import process_pdf_folder
import os
from tkinter import filedialog
from PIL import ImageTk, Image
import shutil
from threading import Thread

remove_additional_files = True


class ConvertionArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GUI(tk.Tk):

    def __init__(self, master=None):
        tk.Tk.__init__(self, master)
        self.title("SSIP 2019 -- Project No.5 -- Scan PDF --")
        self.geometry("900x550")
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
        self.submitButton.grid(row=2, column=2, padx=10)
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
        self.convertTriggerButton = tk.Button(master, text="Convert:", fg="green", font=("bold"), command=self.startThread)
        self.convertTriggerButton.grid(row=10, column=4)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.progress.grid(row=10, column=0)
        self.maxbytes = 100
        self.progress["maximum"] = 100
        # -------------------------------------------------------------------
        # --------------------------------------------------------------------

    def startThread(self):
        t1 = Thread(target=self.doSomething)
        t1.start()

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
            self.lnameLabel = tk.Label(self, text="Select your files", bg='#19cefd')
            self.lnameLabel.grid_forget()
            self.lnameLabel = tk.Label(self, text=self.mainPath, bg='#19cefd')
            self.lnameLabel.grid(row=10, column=5)
        else:
            self.lnameLabel = tk.Label(self, text="Invalid path!", bg='#19cefd')
            self.lnameLabel.grid(row=10, column=5)

    def updateProgressBar(self, value):
        '''simulate reading 500 bytes; update progress bar'''
        self.value = value * 20
        self.progress["value"] = self.value

    # call your functions below:
    # ------------------------------------------------
    def doSomething(self):
        self.progress["value"] = 0
        output_folder = self.mainPath if os.path.isdir(self.mainPath) else os.path.dirname(self.mainPath)
        output_folder = "{}/{}".format(output_folder, self.mainPath.split("/")[-1].split(".")[0])
        args = ConvertionArgs(input_folder=self.mainPath, output_folder=output_folder, work_folder=output_folder + "/work")
        process_pdf_folder(args, self.updateProgressBar)
        if remove_additional_files:
            shutil.rmtree(output_folder + "/work", ignore_errors=True)


if __name__ == "__main__":
    guiFrame = GUI()
    guiFrame.mainloop()
