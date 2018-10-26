# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:09:03 2018

@author: haoqi
"""

import os
directory = "脉象数据20181024\\"
os.chdir(directory) # 改变当前工作目录
cwd = os.getcwd() # 查看当前工作目录
print("--------------current working directory : " + cwd + "----------")

def deleteBySize(minSize):
    """删除小于minSize的文件（单位：K）"""
    files = os.listdir(os.getcwd()) # 列出目录中文件
    for file in files:
    ##    print file + " : " + str(os.path.getsize(file))
        if os.path.getsize(file) < minSize * 1000:
            os.remove(file)
            print(file + " deleted.")
    return

def deleteNullFile():
    '''删除所有大小为0的文件'''
    files = os.listdir(os.getcwd()) # 列出目录中文件
    for file in files:
        if os.path.getsize(file) == 0: #得到文件大小，如果是目录返回0
            os.remove(file)
            print(file + " deleted")
    return

def deleteFile_by_list(list_todelete):
    '''删除指定序号的文件'''
    files = os.listdir(os.getcwd()) # 列出目录中文件
    i = 1
    for file in files:
        if i in list_todelete:
            os.remove(file)
            print(file + ' deleted')
        i += 1
    return