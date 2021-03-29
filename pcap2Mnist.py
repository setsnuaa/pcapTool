#!/usr/bin/env python
# coding: utf-8

# In[1]:


#调用shell
import os
def shell_call(command):
    val=os.system('%s'%(command))
    return val


if __name__=="__main__":
    shell_call("pwsh 1_Pcap2Session.ps1 -s")
    print("pcap to session done--------")
    shell_call("pwsh 2_ProcessSession.ps1 -a [-s]")
    print("process session done--------")
    shell_call("python3 3_Session2Png.py")
    print("session to png done--------")
    shell_call("python3 4_Png2Mnist.py")
    print("png to mnist done--------")
