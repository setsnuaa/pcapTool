**pcap转换工具操作说明**

**1.环境配置**

系统：Ubuntu 20.04 

**安装Mono：**

1）安装必要的软件包：

```
sudo apt update
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
```

2）导入源仓库的GPG key：

```
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
```

3）添加Mono源仓库到系统源列表：

```
sudo apt-add-repository 'deb https://download.mono-project.com/repo/ubuntu stable-bionic main'
```

4）安装Mono：

```
sudo apt install mono-devel
```

安装成功的话，输入`mono --version`会显示版本

**安装PowerShell：**

1）安装Snap：

```sudo apt install snap
sudo apt install snap
```

2）使用Snap安装PowerShell：

```
sudo snap install powershell --classic
```

**安装fdupes：**

```
sudo apt-get install fdupes
```

**numpy和pillow：**

确保python环境中numpy>=1.16.4,pillow>=3.4.2,若没有则使用pip指令安装numpy==1.15.4，pillow==3.4.2

**2.使用pcap转换工具**

在pcap转换工具打开命令行终端，

1）确保需要转换的pcap文件在`1_Pcap`文件夹中，将pcap转换为Mnist数据集格式（数据为idx3格式，标签为idx1格式），数据集保存在`5_Mnist`文件夹中

```
python3 pcap2Mnist.py
```

2)若需要删除pcap转换为Mnist数据集格式所产生的中间文件（这些文件不需要输入模型）

```
python3 removeTmpFiles.py
```

