import tkinter as tk
from tkinter import filedialog
import api


# def select_image():
#     root = tk.Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename()
#     return file_path
#
# image_filepath = select_image()
# test = api.yun_guan_api(image_filepath)
# print(test)


def welcome_interface():
    root = tk.Tk()
    root.title("欢迎使用观云识天")
    root.geometry("300x150")
    tk.Label(root, text="请选择图片文件").pack()
    button_select = tk.Button(root, text="选择图片", command=identify_image)
    button_select.pack()
    tk.Label(root, text="点击按钮，选择图片进行识别").pack()
    root.mainloop()


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def identify_image():
    # 在这里实现上传图片的功能
    image_filepath = select_image()
    test = api.yun_guan_api(image_filepath)
    print(test)
    pass


welcome_interface()

