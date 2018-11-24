plt.plot(performance['train']['loss'])
plt.plot(performance['valid']['loss'])
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)')
plt.title('Training and Validation Loss')
plt.axvline(performance['best_epoch'])
plt.savefig('figures/training_loss.jpg')

