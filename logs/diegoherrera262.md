# DiegoHerrera262's Project Log 2021-1

**Date:** 10/03/21
**Topic:** Variational Circuits

Today I discovered Xanadu's tools for Quantum Machine learning, and got a grasp on the fundamentals of hybrid computing on this area. Also, the name of the packages is pretty fun. I still have not been very acquainted with the full details of QML. However, I can tell that the main principle, according to Xanadu, is to use simple **Quantum Nodes**, that carry out subroutines that are computationally expensive on a classical computer. This may include a feature map or the computation of a cost function.

> The most remarkable feature of Xanadu's interface for NISQ QML is that their software allows *gradient computation* without much effort.

As far as I'm concerned, Xanadu allows simulation of digital processors and optical devices. They have two default simulators: ```default_qubit``` and ```default_gaussian```. For digital and optical devices respectively. Furthermore, they allow usage of other companies' devices. The trick is that user must install so called **plug-in**s or define a custom one. The funniest plug-in is called *StrawberryFields*. This is for continuous variable (optical) computation simulation.

The actual python library is called ```pennylane```. I see that this can be quite useful for learning at prototyping at least. Hence I proceeded to install it. There is some scatetred information, that I intend to summarize here. First, I explain how to install the packages on a MacOS device. Then, I explain how to install ```pennylane-sf``` plug-in for using StrawberryFields. After that, I explain how to interface it with Qiskit using ```pennylane-qiskit```.

**IMPORTANT:** I am using PC with MacOS BigSur 11.2.3. I do not know how to install any of this packages on Windows, but presume installation on a Ubuntu-based device might not be too different.

### Environment setup

To this date, although latest Python version is 3.9.2, I discovered that this version is not compatible con all plug-ins. Therefore, I set up a conda environment with python 3.8. This can be done by the command

```bash
conda create -n QML python=3.8
```

**IMPORTANT:** I suggest using Ananda Python as default distro. This is because it allows easy environment management and ensures compatibility most of the times. This can be downloaded from the [official website](https://www.anaconda.com/products/individual). I suggest using command line installer rather than graphical installer. Sometimes the former may not link path variables properly.

Any time I work on Quantum Machine Learning, I will activate my environment using command

```bash
conda activate -n QML
```

### Installing PennyLane

PennyLane is the Python package built by Xanadu for Quantum Machine Learning. The general structure is that it interfaces NumPy, PyTorch or TensorFlow with a QPU. The QPU performs what are called **Quantum Nodes** or QNodes, which are subroutines where a QPU might give a substantial speed up. As mentioned earlier, the main feature of PennyLane is that it allows quantum computation of **gradients**, which simplifies a lot development of hybrid QML algorithms.

To install this package, I activated my conda environment, and ran the command

```bash
pip install pennylane
```

This did the job for me. I link here the [official page](https://pennylane.ai/install.html) where all the installation instructions can be found.

**NOTE:** I tried installing wrappers for NumPy and PyTorch, but found all requirements were already satisfied by the afore mentioned command.

### Installing StrawberryFields simulator

As mentioned earlier, PennyLane already has two default simulators. However, some tutorials make use of StrawberryFields for continuous variable computations. So I decided to install the plug-in anyway. Plus, I love the name of the packages.

To be able to use the plug-in, I ran the command

```bash
pip install pennylane-sf
```

This did the job. I link here the [installation instructions](https://pennylane-sf.readthedocs.io/en/latest/installation.html) for more details.

### Installing Qiskit Plug-in

The best part to me is that PennyLane can be interfaced with many other quantum computing libraries and services. At this stage, I only use IBM Q. Therefore, I only see necessary to install Qiskit plug-in. This will allow me to simulate QML algorithms with ```qsm_simulator``` from Qiskit Aer, but more important... I will be able to execute them on a real IBM Q device. This step is little tricky. I did what produced no error messages, but haven't tested it yet.

First, install the plug-in pretty much like StrawberryFields

```bash
pip install pennylane-qiskit
```

I already had Qiskit installed on my computer. However, if it were not installed, this command would have resolved dependencies. As is well known, execution of a quantum algorithm in an IBM Q device requires a user account. An IBM Q account has a unique token that is called by Qiskit to run a quantum circuit on a superconducting device. The way PennyLane does this is by a ```config.toml``` file, located at an specific path, which depends on the OS ([see here](https://meet.google.com/edb-vjvo-xix)). On MacOs, this path is

```bash
~/Library/Preferences/pennylane
```

Supposedly, PennyLane installation generates this file. However, I had to create it myself. The content depends on user preferences. The most important part is to include IBM Q token for remote execution. I will not paste all the contents of the file here, but refer [here](https://github.com/carstenblank/pennylane-qiskit) for further information.

**IMPORTANT:** Make sure you have an IBM Q account.

### Installing PyTorch

It is possible that I need PyTorch for some hybrid algorithms. Therefore, I installed it. The documentation can be found [here](https://pytorch.org). The installation was carried out with ```conda```

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

My plan now is to follow PennyLane Tutorials for QML. The first in my list is a simple variational circuit that implements an X gate from two rotations. [Here]() are the tutorials I will follow.
