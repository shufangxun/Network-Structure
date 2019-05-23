import numpy as np

def nms(dets, threshold):
    """ baseline
        dets:(N,5) 
        scores:(N,) = dets[:,4]
        order:(N,)
    """
    x1 = dets[:,0]   # left_down_x
    y1 = dets[:,1]   # left_down_y
    x2 = dets[:,2]   # right_up_x
    y2 = dets[:,3]   # right_up_y
    scores = dets[:,4]  

    order = scores.argsort()[::-1]  # argsort是升序排序的索引,[::-1]得降序的索引
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 当前所有候选框的面积

    keep = [] # 保留的候选框索引

    while order.size > 0:
        i = order[0]  # 最大score的候选框索引
        keep.append(i) 

        # inter 
        # xx1,yy1是左下角;xx2,yy2是右上角 
        # 向量化操作 得到所有候选框与最大框的相交坐标组
        # 当没有相交区域时,为0
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        union = area[i] + area[order[1:]] - inter 
        iou = inter / union
        idx = np.where(iou <= threshold)[0]
        order = order[idx + 1] # 关键所在,因为iou数组比order数组长度小一个,所以idx要+1才匹配
        # 这一步 order变成order[0],order[1]......
    
    return keep

if __name__ == "__main__":
    dets = np.array([[100,120,170,200,0.98],
                     [20,40,80,90,0.99],
                     [20,38,82,88,0.96],
                     [200,380,282,488,0.9],
                     [19,38,75,91, 0.8]])
    print(nms(dets, 0.5))

    