'''
3 plots - make plots for model performance comparison
'''

def makeplots(losses_train, losses_valid, accus_train, accus_valid, model_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_valid, label='Validation Loss')
    plt.title('Training and Validation Loss - ' + model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(accus_train, label='Training Accuracy')
    plt.plot(accus_valid, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy - ' + model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def makeplots2(losses_train, losses_valid, accus_train, accus_valid, model_name):
    ''' Plot training and validation performance in one graph '''
    import matplotlib.pyplot as plt
    
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    # Plot training and validation losses on the left y-axis
    ax1.plot(losses_train, label='Training Loss', color='red')
    ax1.plot(losses_valid, label='Validation Loss', color='blue')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.set_xlabel('Epoch')
    
    # Create a secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(accus_train, label='Training Accuracy', color='coral')
    ax2.plot(accus_valid, label='Validation Accuracy', color='dodgerblue')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')
    
    # Add a grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Title and legend
    fig.suptitle(f'Training and Validation Performance - {model_name}')
    fig.tight_layout()  
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.show()

