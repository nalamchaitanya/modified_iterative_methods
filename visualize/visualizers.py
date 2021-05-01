
from matplotlib import pyplot as plt

def show_l_inf_plot(kind='',dim='',method='',spectral_radius='',l_inf_values=[],s=1):
    plt.scatter(x = range(0,len(l_inf_values)),y = l_inf_values,s=s)
    plt.title(f'Kind : {kind}\nDimension : {dim}\nMethod : {method}\nSpectral radius : {spectral_radius}')
    plt.xlabel('Iteration Number')
    plt.ylabel('L_inf',rotation=0)
    plt.show()

def show_iterations_plot(kind='',dim='',y=[],iteration_values=[],s=1):
    plt.barh(y=y[::-1],width=iteration_values[::-1])
    plt.xlim([0,max(iteration_values)*1.1])
    for index,value in enumerate(iteration_values[::-1]):
        plt.text(value,index,str(value))
    plt.title(f'Kind : {kind}\nDimension : {dim}\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Method')
    plt.show()
    
def show_spectral_radius_plot(kind='',dim='',y=[],spectral_radius_values=[],s=1):
    plt.barh(y=y[::-1],width=spectral_radius_values[::-1])
    plt.xlim([0,max(spectral_radius_values)+1])
    for index,value in enumerate(spectral_radius_values[::-1]):
        plt.text(value,index,str(value))
    plt.title(f'Kind : {kind}\nDimension : {dim}\n')
    plt.ylabel('Method')
    plt.xlabel('Spectral Radius')
    plt.show()