<h2 style='pointer-events: none;'>配置python环境（学习python之前的步骤）</h2>

本教程将指导如何安装 PyCharm 并设置 Conda 环境，以便可以在 PyCharm 中轻松开发和管理 Python 项目。
<h3 style='pointer-events: none;'>步骤 1：安装 PyCharm</h3>

1. 在浏览器中打开 [JetBrains 官方网站](https://www.jetbrains.com/pycharm/)。
2. 下载对应操作系统的 PyCharm 版本（Community版本就行，Professional版本需要付费或者学生认证）。
3. 运行安装程序，并按照屏幕上的说明进行安装。
4. 完成安装后，启动 PyCharm，如果 PyCharm 成功启动并显示欢迎界面，那么安装就成功了。
<h3 style='pointer-events: none;'>步骤 2：下载和安装 Conda</h3>

1. 在浏览器中打开 [Anaconda 官方网站](https://www.anaconda.com/products/individual)。
2. 下载对应操作系统的 Anaconda 版本。
3. 运行安装程序，并按照屏幕上的说明进行安装。
4. 确认 Conda 是否成功安装：在命令行中输入 `conda --version`，如果显示出 Conda 的版本信息，则安装成功。
<h3 style='pointer-events: none;'>步骤 3：设置中文</h3>

1. 在 PyCharm 中，打开 "设置"（Settings）窗口。通过 "文件"（File）菜单中的 "设置"（Settings）选项访问它。
2. 在设置窗口中，选择 "插件"（Plugins）。
3. 在插件管理页面，单击 "市场"（Marketplace）选项卡。
4. 在搜索框中输入 "Chinese Translation"（中文翻译）。
5. 在搜索结果中找到 "Chinese Translation" 插件，并单击 "安装"（Install）按钮。
6. 完成插件安装后，重启 PyCharm。
<h3 style='pointer-events: none;'>步骤 4：创建 Conda 环境并配置项目</h3>

1. 打开 PyCharm，并在欢迎界面上选择 "创建新项目"。
2. 在项目类型列表中选择 "纯 Python"。
3. 在 "位置" 字段中指定项目的文件夹位置，并为项目命名。
4. 在 "解释器" 字段中，单击 "新建环境" 图标。
5. 在弹出窗口中，选择 "Conda 环境" 选项卡。
6. 在 "Conda 可执行文件" 字段中，提供 Conda 可执行文件的路径。如果 Conda 已正确安装并添加到系统路径中，则无需更改此字段。
7. 单击 "确定"。
8. 在 "基本解释器" 字段中，选择要使用的 Python 版本。
9. 在 "环境名称" 字段中，为刚创建的 Conda 环境命名。
10. 单击 "确定"。
11. 在项目设置中，将看到新创建的 Conda 环境被设置为项目的解释器。
12. 单击 "创建" 按钮，完成项目创建过程。
<h3 style='pointer-events: none;'>步骤 5：安装和管理 Python 包</h3>

1. 在 PyCharm 中，打开 "设置"（Settings）窗口。通过 "文件"（File）菜单中的 "设置"（Settings）选项访问它。
2. 在设置窗口中，选择 "项目：[你的项目名]" -> "Python 解释器"。
3. 在 Python 解释器页面，可以看到已经安装的所有 Python 包。
4. 要安装新的 Python 包，单击 "＋" 按钮，然后在弹出的窗口中搜索你需要的包，选择正确的版本后，单击 "安装包" 按钮。
5. 要卸载已经安装的 Python 包，选择你要卸载的包，然后单击 "－" 按钮。
6. 完成后，单击 "应用" 按钮，保存你的更改。
