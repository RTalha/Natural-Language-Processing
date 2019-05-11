#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.22
#  in conjunction with Tcl version 8.6
#    May 09, 2019 10:23:56 PM PKT  platform: Linux

import sys

LARGE_FONT= ("Verdana", 12)
NORM_FONT= ("Verdana", 10)
SMALL_FONT= ("Verdana", 8)

try:
    from Tkinter import *
except ImportError:
    from tkinter import *


try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import gui_support
def helloCallBack():
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text="hello world kindly go to heell", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    label = ttk.Label(popup, text="hello world kindly go to heell", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    gui_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    gui_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'

        top.geometry("600x450+380+100")
        top.title("New Toplevel")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=-0.017, rely=-0.022, relheight=1.033
                , relwidth=1.008)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(width=605)

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.38, rely=0.387, height=59, width=163)
        self.Button2.configure(background="#fff7fc")
        self.Button2.configure(font="-family {nimbus mono l} -size 14")
        self.Button2.configure(foreground="#ce2d53")
        self.Button2.configure(text='''Maing VSM''')
        self.Button2.configure(width=163)

        self.Button2_3 = tk.Button(self.Frame1)
        self.Button2_3.place(relx=0.38, rely=0.194, height=59, width=163)
        self.Button2_3.configure(activebackground="#f93778")
        self.Button2_3.configure(activeforeground="white")
        self.Button2_3.configure(background="#fff7fc")
        self.Button2_3.configure(font="-family {nimbus mono l} -size 14")
        self.Button2_3.configure(foreground="#ce2d53")
        self.Button2_3.configure(text='''PreProcessing''',command=lambda :controller.show_frame(Toplevel2))

        self.Button2_4 = tk.Button(self.Frame1)
        self.Button2_4.place(relx=0.38, rely=0.581, height=59, width=163)
        self.Button2_4.configure(activebackground="#f9f9f9")
        self.Button2_4.configure(background="#fff7fc")
        self.Button2_4.configure(font="-family {nimbus mono l} -size 14")
        self.Button2_4.configure(foreground="#ce2d53")
        self.Button2_4.configure(text='''Classifiers''')

        self.Button2_5 = tk.Button(self.Frame1)
        self.Button2_5.place(relx=0.38, rely=0.774, height=59, width=163)
        self.Button2_5.configure(activebackground="#f9f9f9")
        self.Button2_5.configure(background="#fff7fc")
        self.Button2_5.configure(font="-family {nimbus mono l} -size 14")
        self.Button2_5.configure(foreground="#ce2d53")
        self.Button2_5.configure(text='''Saving Model''')

class Toplevel2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


if __name__ == '__main__':
    vp_start_gui()




