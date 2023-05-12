import matplotlib.pylab as plt
from matplotlib.pyplot import figure
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

cmap = sns.diverging_palette(230, 20)
c=sns.color_palette("rocket")
colors_used = np.array([c[0],'#cc4c02',cmap[1],c[5]])

def plotting_fun2(deaths,total_pops):
    base_case = deaths
            
    w_d=base_case[0:5]
    w_p =total_pops[0:5]
    w_m = w_d/w_p
    b_d=base_case[5:10]
    b_p =total_pops[5:10]
    b_m = b_d/b_p
    
    age_pops=total_pops[0:5]+total_pops[5:10]

    w_m = np.append(w_m,sum(w_m*age_pops/sum(age_pops)))
    b_m = np.append(b_m,sum(b_m*age_pops/sum(age_pops)))

    comp_plot = b_m/w_m
    c = 0
    for i in comp_plot:
        if np.isnan(i):
            comp_plot[c]=1
        if np.isinf(i):
            comp_plot[c]= b_m[c]/(1/w_p[c])
        c =c+1 
    # set width of bar
    barWidth = 0.6
    fig,ax = plt.subplots(figsize =(15, 4))

    # Set position of bar on X axis
    br1 = np.arange(6)
    br2 = [x + barWidth for x in br1]
    c=sns.color_palette("rocket") #0,1,5,4
    # Make the plot
    plt.bar(br1, np.round(comp_plot,1), color =colors_used[1], width = barWidth,
            edgecolor ='grey', label ='Other')

    label = np.round(comp_plot,1)
    bars2 = np.round(comp_plot,1)
    for i in range(len(br1)):
        plt.text(x = br1[i]-0.15 , y = bars2[i]+0.1, s = str(label[i])+'x', fontsize=20, color = 'k')

    plt.title('COVID-19 Minority Mortality Rates Compared to White Population', fontweight ='bold', fontsize = 20)
    plt.xticks([r for r in range(6)],
            ["Ages\n 0-19","Ages\n 20-49","Ages\n 50-59","Ages\n 60-69","Ages\n 70+","Age\n Adjusted"], fontsize = 20)
    ax.axes.yaxis.set_visible(False)
    plt.ylim([0,15])
    plt.show()
    return comp_plot