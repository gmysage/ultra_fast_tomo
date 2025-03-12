from ultra_fast_tomo_ML import *
import matplotlib.pyplot as plt

def script_test():   
    device = 'cuda'
    image = io.imread('./example/img_test.tiff')    
    model = RRDBNet(1, 1, 16, 4, 32)
    model_path = './pre_trained_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    img_processed = check_image_fitting(model, image, device, plot_flag=1, clim=[0,1], ax='off', title='', figsize=(14, 8))
    plt.show()
    
if __name__ == '__main__':
    script_test()
