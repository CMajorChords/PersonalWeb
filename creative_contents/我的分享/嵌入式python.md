<h2 style='pointer-events: none;'>使用python embeddable package分发python程序</h2>

很多诸如Django, Flask, Tornado等web框架都是基于python的，但是这些框架打包起来太麻烦，在一些情况下，我们并不能指望用户会安装python环境直接运行代码，这时候我们可以使用python的embeddable package来分发我们的python程序，然后通过编写bat文件或者shell脚本来启动我们的程序。
<h3 style='pointer-events: none;'>什么是python embeddable package</h3>

python embeddable package是一个精简的python环境，它不包含python的标准库，只包含python的核心库。python embeddable package的意义是直接将python环境打包到我们的程序中，这样我们的程序就不需要依赖用户的python环境了。
<h3 style='pointer-events: none;'>如何使用python embeddable package</h3>

首先使用[python官网](https://www.python.org/downloads/windows/)下载python的embeddable package，如果电脑是64位的，那么就下载64位的，如果是32位的，那么就下载32位的，注意绝大多数的电脑都是64位的，如果不确定，就下载64位的。

下载后解压到一个目录，将该目录下一个叫`pythonxx._pth`（xx为python版本号）的文件打开，该文件默认内容为：
```
python38.zip
.
# Uncomment to run site.main() automatically
#import site
```
首先删去最后一行的注释，即删去`#`，将`#import site`改为`import site`，保存文件。
其次说明一下这段代码的意义，代码中的前两行：
```
python38.zip
.
```
`python38.zip`表示python的核心库，`.`表示当前目录，这两个是必须的，不可删除。如果你编写的程序需要依赖其他的目录，那么可以在这里添加，例如该python环境一般放在项目的子目录下，那么可以添加`..`，表示项目的根目录：
```
python38.zip
.
..
```
网上很多推荐在embeddable package中安装pip的，这里并不推荐，使用pip可能会导致一些问题，如果要安装第三方库，可以使用系统的python所带的pip来安装到embeddable package下专门的目录中。例如如果我们想安装pandas和streamlit这两个库，在python嵌入式环境所在的目录下打开终端，输入：
```
python -m pip install pandas streamlit -t .\Lib\site-packages
```
这样就可以将pandas和streamlit安装到embeddable package的环境中了。要在python嵌入式环境中运行程序，可以使用bat文件或者shell脚本，例如我们的程序是`main.py`，那么可以编写一个`run.bat`文件：
```
@echo off
.\python.exe main.py    
```
然后双击`run.bat`文件就可以运行我们的程序了。
<h3 style='pointer-events: none;'>web框架的使用</h3>

如果我们使用的是web框架，例如streamlit，那么对于streamlit的主文件`app,py`我们可以编写一个`run.bat`文件在项目的根目录下：
```
setlocal
.\environment\python.exe -m streamlit run app.py
end local
```
这段代码的意义是使用项目根目录下的python环境来运行`app.py`文件。`-m`指定了要运行的模块，`streamlit`是streamlit的模块，`run`是streamlit的命令，`app.py`是streamlit的主文件，python环境所在的目录是`environment`。
<h3 style='pointer-events: none;'>软件的打包</h3>
<h4 style='pointer-events: none;'>将bat文件打包成exe文件</h4>

如果我们想将bat文件转化为exe文件，可以使用[Bat To Exe Converter](https://pan.baidu.com/s/1sXiSBxOMkNY61VdaY3BaRw?pwd=Funz)这个软件，这个软件可以将bat文件转化为exe文件，这样我们就可以将我们的程序分发给用户了。
<h4 style='pointer-events: none;'>使用winrar建立自解压程序并发送exe文件到桌面</h4>

如果我们想将我们的程序打包成一个exe文件，然后用户双击exe文件就可以解压到桌面，可以使用winrar，首先将我们的右键选择**添加到压缩文件**，然后勾选**创建自解压格式压缩文件**，选择**高级**，选择**自解压选项**，在**常规**中选择默认的解压路径，然后在**文本和图标**中填写选项，其中**从文件加载自解压文件徽标**和**高精度自解压文件徽标**可以选择一个图标，徽标表示安装程序打开后窗口的背景，**从文件夹加载自解压图标**这一项可以选择一个图标，这个图标表示自解压exe程序的图标。
在设置中还可以编写要运行的程序，该程序将在解压后运行。
