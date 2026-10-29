    const sideWhiteParamFields = [
        ['csjbl_preview_radius_product_px','CSJBL 顯示範圍',50,0,500,1,'產品 px；只影響黃框。'],
        ['min_contrast_gray','最低局部反差',2.2,.1,255,.1,'灰階差；越低越敏感。'],
        ['noise_sigma_factor','雜訊門檻倍率',4.5,.1,100,.1,'倍；越高越能抑制雜訊。'],
        ['min_area_px','最小候選面積',20,1,1000000,1,'側拍像素數；過濾小碎點。'],
        ['edge_margin_px','邊緣排除寬度',16,0,512,1,'側拍 px；0 不內縮。'],
        ['dust_overlap_ratio','灰塵重疊門檻',.8,.01,1,.01,'0～1；候選整體或暗色本體覆蓋率，符合條件即屏蔽。'],
        ['mapping_margin_px','灰塵比對外擴',3,0,32,1,'側拍 px；外擴 OMIT 遮罩。'],
        ['crop_padding_px','組合圖周邊範圍',32,0,512,1,'各原圖 px。'],
        ['bomb_tolerance_product_px','炸彈容許誤差',50,0,500,1,'產品 px；比對正拍炸彈。'],
    ];
    const sideWhiteSwitches = [['apply_exclusions','套用產品不檢測區域',true],['bomb_check_enabled','比對 W0F00000 炸彈座標',true],['bomb_force_detection_enabled','炸彈區域強制檢測',false]];
