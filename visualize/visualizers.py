
from matplotlib import pyplot as plt

def show_l_inf_plot(kind='',dim='',method='',spectral_radius='',l_inf_values=[],s=10):
    plt.scatter(x = range(0,len(l_inf_values)),y = l_inf_values,s=s,c='#0303fc')
    plt.title(f'Kind : {kind}\nDimension : {dim}\nMethod : {method}\nSpectral radius : {spectral_radius}')
    plt.xlabel('Iteration Number')
    plt.ylabel('||X-X*||_inf')
    plt.show()

def show_iterations_plot(kind='',dim='',y=[],iteration_values=[],s=1,color='blue'):
    plt.barh(y=y[::-1],width=iteration_values[::-1],color=color)
    plt.xlim([min(iteration_values)*0.9,max(iteration_values)*1.1])
    for index,value in enumerate(iteration_values[::-1]):
        plt.text(value,index,str(round(value,6)))
    plt.title(f'Kind : {kind}\nDimension : {dim}\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Method')
    plt.show()
    
def show_spectral_radius_plot(kind='',dim='',y=[],spectral_radius_values=[],s=1,color='blue'):
    plt.barh(y=y[::-1],width=spectral_radius_values[::-1],color=color)
    width = max(spectral_radius_values) - min(spectral_radius_values)
    plt.xlim([min(spectral_radius_values)-width/2,max(spectral_radius_values)+width/2])
    for index,value in enumerate(spectral_radius_values[::-1]):
        plt.text(value,index,str(round(value,6)))
    plt.title(f'Kind : {kind}\nDimension : {dim}\n')
    plt.ylabel('Method')
    plt.xlabel('Spectral Radius')
    plt.show()