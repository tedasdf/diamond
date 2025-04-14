import serial
from tkinter import *
import time

last_time1 = 0
last_time2 = 0
last_time3 = 0
last_time4 = 0

root = Tk()
root.title("Controlling robotic arm")




def send1(n):
    global last_time1
    n = int(n)+1000
    precise_time1 = time.time()
    if last_time1+1< precise_time1:
        ser.write(bytes(str(n), 'utf-8'))
        print(n)
        last_time1 = precise_time1

def send2(n):
    global last_time2
    n = int(n)+2000
    precise_time2 = time.time()
    if last_time2+1< precise_time2:
        ser.write(bytes(str(n), 'utf-8'))
        print(n)
        last_time2 = precise_time2
def send3(n):
    global last_time3
    n = int(n)+3000
    precise_time3 = time.time()
    if last_time3+1< precise_time3:
        ser.write(bytes(str(n), 'utf-8'))
        print(n)
        last_time3 = precise_time3
def send4(n):
    global last_time4
    n = int(n)+4000
    precise_time4 = time.time()
    if last_time4+1< precise_time4:
        ser.write(bytes(str(n), 'utf-8'))
        print(n)
        last_time4 = precise_time4

def computer():
    ser.write(bytes(str(2),'utf-8'))
def potenciometer():
    ser.write(bytes(str(1), 'utf-8'))


ser = serial.Serial('COM3', 9600, timeout= 50,write_timeout = 50)
computer()
label_1 = Label(root, text = "Choose control")
label_1.grid(row = 0, column = 0,sticky= W)
var = IntVar()
var.set(1)

slider1  = Scale(root,from_ =0,to = 180,orient = VERTICAL, length = 500, command =send1)
slider2  = Scale(root,from_ =0,to = 180,orient = VERTICAL, length = 500, command =send2)
slider3  = Scale(root,from_ =0,to = 180,orient = VERTICAL, length = 500, command =send3)
slider4  = Scale(root,from_ =0,to = 180,orient = VERTICAL, length = 500, command =send4)

b = Radiobutton(root, text="Potentiometer",variable=var, value=1,command =potenciometer)
b.grid(row = 1, column = 0,sticky= W)
b = Radiobutton(root, text="Computer",variable=var, value=2,command =computer)
b.grid(row = 2, column = 0,sticky= W)

slider1.grid(row = 0, column = 1,rowspan = 100)
slider2.grid(row = 0, column = 2,rowspan = 100)
slider3.grid(row = 0, column = 4,rowspan = 100)
slider4.grid(row = 0, column = 6,rowspan = 100)

root.mainloop()
