import matplotlib.pyplot as plt
import cv2

def visualize_merged_amodal_and_modal(merged_image, amodal_mask, modal_mask):
    """
    합성된 이미지, Amodal 마스크, Modal 마스크를 시각화하는 함수
    :param merged_image: 합성된 이미지 (RGB)
    :param amodal_mask: Amodal 마스크 (오이 전체 마스크)
    :param modal_mask: Modal 마스크 (잎에 의해 가려진 오이 마스크)
    """
    
    # Plot 설정
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))  # 1행 3열 레이아웃
    
    # 합성 이미지
    axs[0].imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환
    axs[0].set_title('Merged Image')
    axs[0].axis('off')

    # Amodal 마스크
    axs[1].imshow(amodal_mask, cmap='gray')
    axs[1].set_title('Amodal Mask')
    axs[1].axis('off')

    # Modal 마스크
    axs[2].imshow(modal_mask, cmap='gray')
    axs[2].set_title('Modal Mask')
    axs[2].axis('off')

    plt.subplots_adjust(wspace=0.1)  # 가로 간격을 0.1로 줄이기
    plt.tight_layout()
    plt.show()

def visualize_all_masks(cucumber_mask, leaf_mask, overlap_mask):
    
    plt.figure(figsize=(10, 5))
    
    # 오이 마스크 시각화
    plt.subplot(1, 3, 1)
    plt.imshow(cucumber_mask, cmap='gray')
    plt.title("Cucumber Mask")
    
    # 잎 마스크 시각화
    plt.subplot(1, 3, 2)
    plt.imshow(leaf_mask, cmap='gray')
    plt.title("Leaf Mask")
    
    # 겹치는 부분(overlap) 시각화
    plt.subplot(1, 3, 3)
    plt.imshow(overlap_mask, cmap='gray')
    plt.title("Modal Mask")
    
    plt.tight_layout()
    plt.show()