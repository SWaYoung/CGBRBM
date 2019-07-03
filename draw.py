import numpy as np
import matplotlib.pyplot as plt

#1
#D_qpsk = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#D_8qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
#D_16qam = [1,1,1,1,1,1,1,1,1,0.994,0.959,0.95,0.879,0,0,0]
#D_32qam = [1,1,1,1,1,1,1,1,1,1,1,0.975,0.312,0,0,0]
#D_64qam = [1,1,1,1,1,1,1,1,1,1,1,1,0.599,0,0,0]
#
#I_qpsk = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#I_8qam_ = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
#I_16qam = [1,1,1,1,1,1,1,1,1,0.994,0.959,0.95,0.879,0,0,0]
#I_32qam = [1,1,1,1,1,1,1,1,1,1,1,0.975,0.312,0,0,0]
#I_64qam = [1,1,1,1,1,1,1,1,1,1,1,1,0.599,0,0,0]

#2
#D_16qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.3333333]
#D_32qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.533333]
#D_64qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.866667]
#
#I_16qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.3333333]
#I_32qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.533333]
#I_64qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.866667]

#3
D_qpsk = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
D_8qam_ = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
D_16qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
D_32qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
D_64qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

I_qpsk = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
I_8qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.97,0.9]
I_16qam = [1,1,1,1,1,1,1,1,1,1,1,1,1,0.87,0.7,0.37]
I_32qam = [1,1,1,1,1,1,1,1,1,1,1,0.97,0.98,0.96,0.79,0.2]
I_64qam = [1,1,1,1,1,1,1,1,1,1,1,0.99,1,0.89,0.63,0.54]

x = [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
x_range = np.arange(25,10)

plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.size"] = "12"

#1
#D_qpsk_line = plt.plot(x,D_qpsk,label='Discrimination-QPSK',color='gold', marker='o')
#D_8qam_line = plt.plot(x,D_8qam,label='Discrimination-8QAM',color='green', marker='v')
#D_16qam_line = plt.plot(x,D_16qam,label='Discrimination-16QAM',color='dodgerblue', marker='s')
#D_32qam_line = plt.plot(x,D_32qam,label='Discrimination-32QAM',color='purple', marker='p')
#D_64qam_line = plt.plot(x,D_64qam,label='Discrimination-64QAM',color='red', marker='x')
#
#I_qpsk_line = plt.plot(x,I_qpsk,label='Identification-QPSK',color='gold', marker='o', linestyle=':')
#I_8qam_line = plt.plot(x,I_8qam_,label='Identification-8QAM',color='green', marker='v', linestyle=':')
#I_16qam_line = plt.plot(x,I_16qam,label='Identification-16QAM',color='dodgerblue', marker='s', linestyle=':')
#I_32qam_line = plt.plot(x,I_32qam,label='Identification-32QAM',color='purple', marker='p', linestyle=':')
#I_64qam_line = plt.plot(x,I_64qam,label='Identification-64QAM',color='red', marker='x', linestyle=':')

#2
#D_16qam_line = plt.plot(x,D_16qam,label='Discrimination-16QAM',color='dodgerblue', marker='s')
#D_32qam_line = plt.plot(x,D_32qam,label='Discrimination-32QAM',color='purple', marker='p')
#D_64qam_line = plt.plot(x,D_64qam,label='Discrimination-64QAM',color='red', marker='x')
#
#I_16qam_line = plt.plot(x,I_16qam,label='Identification-16QAM',color='dodgerblue', marker='s', linestyle=':')
#I_32qam_line = plt.plot(x,I_32qam,label='Identification-32QAM',color='purple', marker='p', linestyle=':')
#I_64qam_line = plt.plot(x,I_64qam,label='Identification-64QAM',color='red', marker='x', linestyle=':')

#3
D_qpsk_line = plt.plot(x,D_qpsk,label='Discrimination-QPSK',color='gold', marker='o')
D_8qam_line = plt.plot(x,D_8qam_,label='Discrimination-8QAM',color='green', marker='v')
D_16qam_line = plt.plot(x,D_16qam,label='Discrimination-16QAM',color='dodgerblue', marker='s')
D_32qam_line = plt.plot(x,D_32qam,label='Discrimination-32QAM',color='purple', marker='p')
D_64qam_line = plt.plot(x,D_64qam,label='Discrimination-64QAM',color='red', marker='x')

I_qpsk_line = plt.plot(x,I_qpsk,label='Identification-QPSK',color='gold', marker='o', linestyle=':')
I_8qam_line = plt.plot(x,I_8qam,label='Identification-8QAM',color='green', marker='v', linestyle=':')
I_16qam_line = plt.plot(x,I_16qam,label='Identification-16QAM',color='dodgerblue', marker='s', linestyle=':')
I_32qam_line = plt.plot(x,I_32qam,label='Identification-32QAM',color='purple', marker='p', linestyle=':')
I_64qam_line = plt.plot(x,I_64qam,label='Identification-64QAM',color='red', marker='x', linestyle=':')

plt.title('Discrimination and Identification')
plt.xlabel('OSNR')
plt.ylabel('Overall Accuracy')
plt.legend(loc='best')
plt.grid(linestyle=':', linewidth=0.5)
plt.show()

