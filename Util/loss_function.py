"""
Author: YidaChen
Time is: 2023/9/5
this Code: 损失函数计算：传入待计算损失的特征，用于计算损失然后返回
"""
import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FusionLoss:
    def __init__(self, model_dict, registration=True, loss_weight=[1, 1, 1, 1, 1, 1, 1, 1]):
        if registration:
            # 红外光：私有特征、共有特征、解码特征
            self.ir_detail_feature = model_dict['ir_detail_features']
            self.ir_global_feature = model_dict['ir_global_features']
            # self.ir_decoder_feature = model_dict['ir_decoder_features']

            # 可见光: 私有特征、共有特征、解码特征
            self.vis_detail_feature = model_dict['vis_detail_features']
            self.vis_global_feature = model_dict['vis_global_features']
            # self.vis_decoder_feature = model_dict['vis_decoder_features']

            # 红外与可见光拼接特征
            self.ir_concat_features = model_dict['ir_concat_features']
            self.vis_concat_features = model_dict['vis_concat_features']
            self.fusion_features = model_dict['fusion_features']

            # 红外与可见光输入图像、融合结果-融合图像
            self.ir_img = model_dict['ir_img']
            self.vis_img = model_dict['vis_img']
            self.fus_img = model_dict['fus_img']

            # 重建损失
            # self.recon_img = model_dict['recon_img']

            # 重建梯度损失
            self.recon_gradient = model_dict['recon_gradient']


        else:
            self.ir_img = model_dict['ir_img']
            self.vis_img = model_dict['vis_img']
            # 变形的红外图像
            self.move_ir_img = model_dict['move_ir_img']

            # 变形场
            # self.deformation_field = model_dict['deform_field']

            # 经过变形场后的变形特征
            self.deformabled_features = model_dict['deformabled_features']

            # 对齐的特征作为变形特征的目标和约束
            self.alignment_features = model_dict['alignment_features']

            # 未配准的融合图像:key-{'fix_fu_img', 'move_fus_img'}
            self.unreg_fus_img = model_dict['fus_img']

        # 是否为配准模型
        self.registration = registration

        # 各个损失的权重系数
        self.loss_weight = loss_weight

    def get_loss(self):
        """
        类的主函数，用于调用其他损失函数，然后总损失
        :return:
        """
        # 如果配准图像对
        if self.registration:
            # # 解码特征的一致性损失
            # ir_decoder_loss = self.private_decoded_align_loss(self.ir_detail_feature, self.ir_decoder_feature)
            # vis_decoder_loss = self.private_decoded_align_loss(self.vis_detail_feature, self.vis_decoder_feature)
            # decoder_loss = ir_decoder_loss + vis_decoder_loss
            #
            # # 对比损失
            # ir_contrastive_loss = self.ir_vis_contrastive_loss(shared_feature1=self.ir_global_feature,
            #                                                    shared_feature2=self.vis_global_feature,
            #                                                    private_feature1=self.ir_decoder_feature)
            # vis_contrastive_loss = self.ir_vis_contrastive_loss(shared_feature1=self.vis_global_feature,
            #                                                     shared_feature2=self.ir_global_feature,
            #                                                     private_feature1=self.vis_decoder_feature)
            # ir_vis_contrastive_loss = ir_contrastive_loss + vis_contrastive_loss
            #
            # # 拼接的特征一致性损失
            # concat_feature_consistent_loss = self.fusion_feature_consistent_loss(self.ir_concat_features,
            #                                                                      self.vis_concat_features)
            #
            # # 特征推远损失
            # ir_push_loss = self.fusion_feature_push_away_loss(fusion_feature=self.ir_concat_features,
            #                                                   private_feature=self.ir_detail_feature,
            #                                                   common_feature=self.ir_global_feature)
            # vis_push_loss = self.fusion_feature_push_away_loss(fusion_feature=self.vis_concat_features,
            #                                                    private_feature=self.vis_detail_feature,
            #                                                    common_feature=self.vis_global_feature)
            # ir_vis_push_loss = ir_push_loss + vis_push_loss
            #
            # # 融合图像损失
            # fusion_loss = self.fusion_img_loss(ir_img=self.ir_img, vis_img=self.vis_img, fus_img=self.fus_img)
            #
            # # 总损失
            # loss = decoder_loss + ir_vis_contrastive_loss + concat_feature_consistent_loss + ir_vis_push_loss + fusion_loss

            # 解码特征的一致性损失
            # loss1 = self.private_decoded_align_loss(ir_private_feature=self.ir_detail_feature, vis_private_feature=self.vis_detail_feature, ir_decode_feature=self.ir_decoder_feature, vis_decode_feature=self.vis_decoder_feature)

            # 共有特征、私有特征：构造2对对比损失
            # loss2 = self.ir_vis_contrastive_loss(ir_global_feature=self.ir_global_feature, ir_private_feature=self.ir_detail_feature, vis_global_feature=self.vis_global_feature, vis_private_feature=self.vis_detail_feature)

            # 融合特征相似度损失
            # loss3 = self.fusion_feature_consistent_loss(ir_fusion_feature=self.ir_concat_features, vis_fusion_feature=self.vis_concat_features)

            # 新增重建损失
            # loss4 = self.reconstruction_img_loss(recon_img=self.recon_img, ir_img=self.ir_img, vis_img=self.vis_img)

            # 新增图像梯度损失
            loss1 = self.reconstruction_img_gradient_loss(recon_gradient=self.recon_gradient, fus_img=self.fus_img, ir_img=self.ir_img, vis_img=self.vis_img)

            # 仅计算融合损失
            loss2 = self.fusion_img_loss(ir_img=self.ir_img, vis_img=self.vis_img, fus_img=self.fus_img)
            # loss2 = self.fusion_img_loss(ir_img=self.recon_gradient['recon_ir_gradient'].detach(), vis_img=self.recon_gradient['recon_vis_gradient'].detach(), fus_img=self.fus_img)

            # loss = loss1["loss"] + loss2["loss"] + loss3["loss"] + loss4["loss"] + loss5["loss"] + loss6["loss"]
            loss = loss1["loss"] + loss2["loss"]

            # print(f"总损失为{loss}")
            return {"loss": loss, "loss1_info": loss1["info"], "loss2_info": loss2["info"]}


        # 未配准图像对
        else:
            # 变形场平滑损失损失
            # loss1 = self.unreg_deformation_field_smoothing_loss(self.deformation_field)

            # 变形特征一致性损失
            loss1 = self.unreg_feature_consistency_loss(feature_alignment=self.alignment_features,
                                                        feature_un_alignment=self.deformabled_features)
            # # 融合特征一致性损失
            # loss2 = self.unreg_fusion_feature_consistency_loss(self.alignment_features[1],
            #                                                    self.deformabled_features[1])
            # # 融合图像一致性损失
            loss2 = self.unreg_fusion_img_consistency_loss(
                fusion_img_alignment=self.unreg_fus_img['fix_fus_img'], fusion_img_un_alignment=self.unreg_fus_img['move_fus_img'])

            # 总损失
            loss = loss1['loss'] + loss2['loss']
            return {"loss": loss, "loss1_info": loss1["info"], "loss2_info": loss2["info"]}
            # loss = loss1['loss']
            # return {"loss": loss, "loss1_info": loss1["info"]}

    def private_decoded_align_loss(self, ir_private_feature, vis_private_feature, ir_decode_feature, vis_decode_feature):
        """
        l2损失，进行强约束：私有特征和解码特征拉进
        :param ir_private_feature:
        :param vis_private_feature:
        :param ir_decode_feature:
        :param vis_decode_feature:
        :return:
        """
        ir_private_feature, vis_private_feature, ir_decode_feature, vis_decode_feature = ir_private_feature.to(device), vis_private_feature.to(device), ir_decode_feature.to(device), vis_decode_feature.to(device)
        # loss1 = F.mse_loss(ir_decode_feature, vis_private_feature)
        # loss2 = F.mse_loss(vis_decode_feature, ir_private_feature)
        # loss1 = F.l1_loss(ir_decode_feature, vis_private_feature)
        # loss2 = F.l1_loss(vis_decode_feature, ir_private_feature)
        # print(f"红外解码特征-可见光私有特征一致性损失： {loss1} 可见光解码特征-红外私有特征一致性损失：{loss2}")
        # ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        loss1 = F.mse_loss(ir_decode_feature, vis_private_feature)
        loss2 = F.mse_loss(vis_decode_feature, ir_private_feature)

        cosine_ir_de = pearsonr(ir_decode_feature.flatten(), vis_private_feature.flatten())
        cosine_vis_de = pearsonr(vis_decode_feature.flatten(), ir_private_feature.flatten())

        loss = (loss1 + loss2) * self.loss_weight[3]
        return {"loss": loss, "info": f"红外解码特征-可见光私有特征一致性损失：loss = {loss.item()}_{loss1 * self.loss_weight[3]}_cc={cosine_ir_de.item()} 可见光解码特征-红外私有特征一致性损失：{loss2 * self.loss_weight[3]}_cc={cosine_vis_de.item()}"}

    def ir_vis_contrastive_loss(self, ir_global_feature, vis_global_feature, ir_private_feature, vis_private_feature, margin=10):
        """
        构造两队对比损失用于拉进共有特征、推远私有特征
        :param anchor: 当前模态的公有特征
        :param positive: 另外一个模态的公有特征
        :param negative: 当前模态的私有特征
        :param margin: 边界阈值
        :return:
        """
        # # 用均方误差来定义距离
        # loss_mse = nn.MSELoss()
        # # 正样本度量
        # dis_anchor_positive = F.l1_loss(ir_global_feature, vis_global_feature)
        # # 负样本度量
        # # 红外光
        # dis_anchor_negative_ir = F.l1_loss(ir_global_feature, ir_private_feature)
        # # 可见光
        # dis_anchor_negative_vis = F.l1_loss(vis_global_feature, vis_private_feature)
        # # 第一种：三元组对比损失，利用relu来替代 max（f, 0）
        # # print(dis_anchor_positive.item(), dis_anchor_negative_ir.item(), dis_anchor_negative_vis.item())
        # # loss1 = torch.relu(dis_anchor_positive - dis_anchor_negative_ir + margin)
        # # loss2 = torch.relu(dis_anchor_positive - dis_anchor_negative_vis + margin)
        # # loss = loss1 + loss2
        #
        # # 第二种：优化对比损失: 不要margin自适应动态调整
        # t = 0.07
        # loss1 = -torch.log(torch.exp(dis_anchor_positive / t) / (torch.exp(dis_anchor_positive / t) + torch.exp(dis_anchor_negative_ir / t)))
        # loss2 = -torch.log(torch.exp(dis_anchor_positive / t) / (torch.exp(dis_anchor_positive / t) + torch.exp(dis_anchor_negative_vis / t)))
        # # 通过正样本距离的倒数和负样本距离的比率来约束
        # epsilon = 1e-6  # 防止除以零
        # # relative_distance_loss = 1 / (dis_anchor_negative_ir / (dis_anchor_positive + epsilon) + dis_anchor_negative_vis / (dis_anchor_positive + epsilon))
        # # loss = (loss1 + loss2 + relative_distance_loss) * self.loss_weight[4]
        # loss = (loss1 + loss2) * self.loss_weight[4]
        #
        # # print(f"anchor 红外光-对比损失： {loss1} anchor 可见光-对比损失：{loss2}")
        # return {"loss": loss, "info": f"{dis_anchor_positive.item() * self.loss_weight[4], dis_anchor_negative_ir.item() * self.loss_weight[4], dis_anchor_negative_vis.item() * self.loss_weight[4]}\nanchor 红外光-对比损失： {loss1.item()} anchor 可见光-对比损失：{loss2.item()}"}
        epsilon = 1e-6  # 防止除以零

        # ssim = kornia.losses.SSIMLoss(11, reduction='mean')

        # cosine_global_loss = ssim(ir_global_feature, vis_global_feature)
        # cosine_ir_private_loss = ssim(ir_global_feature, ir_private_feature)
        # cosine_vis_private_loss = ssim(vis_global_feature, vis_private_feature)
        # cosine_private_loss = ssim(ir_private_feature, vis_private_feature)
        cosine_global_loss = pearsonr(ir_global_feature.flatten(), vis_global_feature.flatten())
        cosine_private_loss = pearsonr(ir_private_feature.flatten(), vis_private_feature.flatten())

        # loss = (cosine_ir_private_loss + cosine_vis_private_loss) / (cosine_global_loss + epsilon)
        loss = ((cosine_private_loss ** 2) / (cosine_global_loss + 1.01)) * self.loss_weight[4]
        # return {"loss": loss, "info": f"公有特征损失:{cosine_global_loss.item()} 红外私有特征损失：{cosine_ir_private_loss.item()} 可见光私有特征损失：{cosine_vis_private_loss.item()}"}
        return {"loss": loss, "info": f"对比损失: loss = {loss.item()}_公有特征相似系数损失：{cosine_global_loss.item()}_私有特征相似系数：{cosine_private_loss.item()}"}

    def fusion_feature_consistent_loss(self, ir_fusion_feature, vis_fusion_feature):
        """
        保证特征间的相似度，使用弱约束L1：红外与可见光融合特征进行一致性约束
        :param ir_fusion_feature:
        :param vis_fusion_feature:
        :return:
        """
        ir_fusion_feature, vis_fusion_feature = ir_fusion_feature, vis_fusion_feature
        ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        fusion_feature_loss = ssim(ir_fusion_feature, vis_fusion_feature)
        loss1 = fusion_feature_loss * self.loss_weight[5]

        cc = pearsonr(ir_fusion_feature.flatten(), vis_fusion_feature.flatten())
        loss2 = (1 - cc) * self.loss_weight[5]

        loss3 = F.l1_loss(ir_fusion_feature, vis_fusion_feature) * self.loss_weight[5]
        loss = (loss1 + loss2 + loss3)
        return {"loss": loss, "info": f"融合特征一致性损失：loss = {loss.item()} 融合特征ssim损失：{fusion_feature_loss.item()}_特征相似系数：{cc.item()}_mse损失：{loss3.item()}"}

        # cosine_sim = F.cosine_similarity(ir_fusion_feature, vis_fusion_feature)
        # loss = (1 - cosine_sim.mean()) * self.loss_weight[5]
        # loss = cosine_sim * self.loss_weight[5]
        # loss = F.l1_loss(ir_fusion_feature, vis_fusion_feature) * self.loss_weight[5]
        # loss = F.l1_loss(ir_fusion_feature, vis_fusion_feature)
        # loss = F.mse_loss(ir_fusion_feature, vis_fusion_feature)
        # print(f"fusion_feature_consistent_loss :{loss.item()}")

        # return {"loss": loss, "info": f"融合特征相似度为:{cc.item()}"}

    def fusion_feature_push_away_loss(self, fusion_feature, private_feature, common_feature):
        """
        融合特征推远损失，共有和私有特征与融合特征差异性越大，代表融合后的特征能更好的融合另外一个模态的数据;此时，融合特征前面的特征层冻结
        :param fusion_feature:
        :param private_feature:
        :param common_feature:
        :return:
        """
        ir_vis_cat_feature = private_feature + common_feature
        # cosine_sim = F.cosine_similarity(ir_vis_cat_feature, fusion_feature)
        cosine_sim = F.cosine_similarity(ir_vis_cat_feature, fusion_feature)
        cosine_sim = torch.mean(cosine_sim)
        loss = 1 + cosine_sim
        # print(f"fusion_feature_push_away_loss :{loss.item()}")

        return loss * 0

    def reconstruction_img_loss(self, recon_img, ir_img, vis_img):
        recon_ir, recon_vis, de_recon_ir, de_recon_vis = recon_img['recon_ir'], recon_img['recon_vis'], recon_img['de_recon_ir'], recon_img['de_recon_vis']
        ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        loss_l1 = F.l1_loss(ir_img, recon_ir) + F.l1_loss(vis_img, recon_vis) + F.l1_loss(ir_img, de_recon_ir) + F.l1_loss(vis_img, de_recon_vis)
        loss_l2 = ssim(ir_img, recon_ir) + ssim(vis_img, recon_vis) + ssim(ir_img, de_recon_ir) + ssim(vis_img, de_recon_vis)
        # loss_l1_de = F.l1_loss(ir_img, de_recon_ir) + F.l1_loss(vis_img, de_recon_vis)
        # loss = (loss_l1 + loss_l1_de) * self.loss_weight[7]
        loss = (loss_l1 + loss_l2) * self.loss_weight[7]
        # return {"loss": loss, "info": f"图像的重建损失为:{loss.item()}: {loss_l1.item()}_{loss_l1_de.item()}"}
        return {"loss": loss, "info": f"图像公有+私有重建损失：loss = {loss.item()} 图像的重建损失为:{loss.item()}_l1:{loss_l1}_ssim:{loss_l2}"}

    def reconstruction_img_gradient_loss(self, recon_gradient, fus_img, ir_img, vis_img):
        recon_ir_gradient = recon_gradient['recon_ir_gradient']
        recon_vis_gradient = recon_gradient['recon_vis_gradient']
        # 求取梯度
        # fus_canny = self.sobel_conv(fus_img)
        # 得到的梯度要一致加上ssim损失
        # ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        # loss = (F.l1_loss(recon_ir_gradient, fus_canny) + F.l1_loss(recon_vis_gradient, fus_canny)) * 0
        l1 = F.l1_loss(recon_ir_gradient, ir_img)
        l2 = F.l1_loss(recon_vis_gradient, vis_img)
        # l3 = F.l1_loss(recon_ir_gradient, recon_vis_gradient)
        # loss = (l1 + l2) * self.loss_weight[6]
        loss = (l1 + l2) * self.loss_weight[3]
        return {"loss": loss, "info": f"融合红外/可见光特征重建图像损失：loss = {loss.item()} 融合图像的一致性损失为:{loss.item()}_{l1}_{l2}"}

    def fusion_img_loss(self, ir_img, vis_img, fus_img):
        """
        计算融合图像与输入红外与可见光光的梯度、结构相似度、像素损失，如果三通道不行后期实验仅用 Y 通道进行
        :param ir_img:
        :param vis_img:
        :param fus_img:
        :return:
        """
        ir_img = ir_img.to(device)
        vis_img = vis_img.to(device)
        fus_img = fus_img.to(device)
        img_max = torch.max(vis_img, ir_img)

        # 像素级损失
        loss_pix = F.mse_loss(fus_img, img_max)
        # loss_pix = F.l1_loss(fus_img, img_max)

        # loss_pix = F.mse_loss(fus_img, ir_img) + F.mse_loss(fus_img, vis_img)
        # loss_pix = 0.5 * (torch.mean(torch.square(fus_img-ir_img)) + torch.mean(torch.square(fus_img-vis_img)))

        # 结构相似性损失
        # mean已经求均值了，而且ssim函数是(1-ssim)/2
        ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        # 使用2个结构相似度之和，原来使用的是与max做结构相似度
        loss_ssim = ssim(fus_img, ir_img) + ssim(fus_img, vis_img)
        # loss_ssim = ssim(fus_img, img_max)
        # loss_ssim = ssim(fus_img, img_max) + ssim(fus_img, ir_img)


        # 梯度损失 - > 使用 canny 算子  mse-> l1   -> fus ir\vis
        # ir_canny = kornia.filters.canny(ir_img)[0]
        # vis_canny = kornia.filters.canny(vis_img)[0]
        # fus_canny = kornia.filters.canny(fus_img)[0]
        # 梯度损失 -> 使用sobel算子
        # ir_canny = self.sobel_conv(ir_img)
        # vis_canny = self.sobel_conv(vis_img)
        fus_canny = self.sobel_conv(fus_img)
        # max_canny = torch.max(ir_canny, vis_canny)
        max_canny = self.sobel_conv(img_max)

        # loss_grad = F.mse_loss(fus_canny, max_canny)
        loss_grad = F.l1_loss(fus_canny, max_canny)

        # 损失权重
        alpha = self.loss_weight

        # 总损失
        loss = alpha[0] * loss_pix + alpha[1] * loss_ssim + alpha[2] * loss_grad
        # loss = loss_pix + loss_ssim + loss_grad

        return {"loss": loss, "info": f"图像融合损失：loss = {loss.item()} 像素级损失：{alpha[0] * loss_pix.item()} 结构相似度损失：{alpha[1] * loss_ssim.item()}  梯度损失：{alpha[2] * loss_grad.item()}..."}

    def unreg_deformation_field_smoothing_loss(self, deformation_field):
        """
        变形场平滑损失，基于 L1 损失约束变形场让它更加平滑
        :param deformation_field:
        :return:
        """
        D1 = deformation_field[0]
        D2 = deformation_field[1]
        D3 = deformation_field[2]

        zero_feature = torch.zeros_like(D1)
        loss1 = F.l1_loss(D1, zero_feature)

        zero_feature = torch.zeros_like(D2)
        loss2 = F.l1_loss(D2, zero_feature)

        zero_feature = torch.zeros_like(D3)
        loss3 = F.l1_loss(D3, zero_feature)

        loss = (loss1 + loss2 + loss3) * 0
        return {'loss': loss, 'info': f"loss = {loss.item()} unreg_deformation_field_smoothing_loss :{loss.item()}"}

    def unreg_feature_consistency_loss(self, feature_alignment, feature_un_alignment):
        """
        配准对齐的特征和未配准未对齐的特征进行特征一致性约束，目的就是让未对齐特征与对齐特征靠齐
        :param feature_alignment:
        :param feature_un_alignment:
        :return:
        """
        ir_concat_features = feature_alignment['ir_concat_features']
        reg_ir_concat_features = feature_un_alignment['ir_concat_features']

        loss = F.l1_loss(ir_concat_features, reg_ir_concat_features)

        return {'loss': loss, 'info': f"loss = {loss.item()} unreg_ir_feature_consistency_loss :{loss.item()}"}

    def unreg_fusion_feature_consistency_loss(self, fusion_feature, unreg_fusion_feature):
        """
        图像融合模块输出的融合特征也应该进行一致性损失约束；是否好用后期实验效果证明
        :param fusion_feature:
        :param unreg_fusion_feature:
        :return:
        """
        ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        loss1 = F.l1_loss(fusion_feature, unreg_fusion_feature)
        loss2 = ssim(fusion_feature, unreg_fusion_feature)
        loss = (loss1 + loss2) * 0
        # print(f"unreg_fusion_feature_consistency_loss :{loss.item()}")
        return {'loss': loss, 'info': f"loss = {loss.item()} unreg_fusion_feature_consistency_loss :{loss.item()}: {loss1.item()}_{loss2.item()}"}

    def unreg_fusion_img_consistency_loss(self, fusion_img_alignment, fusion_img_un_alignment):
        """
        配准融合图像作为标准，让未配准图像进行对齐;融合的特征是否还需要加一个损失约束
        :param fusion_img_alignment:
        :param fusion_img_un_alignment:
        :return:
        """

        # ir_img = self.ir_img.to(device)
        # vis_img = self.vis_img.to(device)
        # un_fus_img = fusion_img_un_alignment.to(device)
        # max_img = torch.max(ir_img, vis_img)

        loss_pix = F.mse_loss(fusion_img_alignment, fusion_img_un_alignment)
        # # 结构相似性损失
        # # mean已经求均值了，而且ssim函数是(1-ssim)/2
        ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        # # 使用2个结构相似度之和，原来使用的是与max做结构相似度
        loss_ssim = ssim(fusion_img_alignment, fusion_img_un_alignment)
        #
        # # 梯度损失 - > 使用 canny 算子  mse-> l1   -> fus ir\vis
        # fus_canny = self.sobel_conv(un_fus_img)
        # ir_canny = self.sobel_conv(ir_img)
        # vis_canny = self.sobel_conv(vis_img)
        # max_canny = torch.max(ir_canny, vis_canny)

        fus_canny = self.sobel_conv(fusion_img_alignment)
        max_canny = self.sobel_conv(fusion_img_un_alignment)
        loss_grad = F.mse_loss(fus_canny, max_canny)

        loss = loss_pix + 10 * loss_ssim + 10 * loss_grad
        # loss = loss_pix
        return {'loss': loss, 'info': f"loss = {loss.item()} unreg_fusion_img_consistency_loss :{loss.item()}_{loss_pix.item()}_{loss_ssim.item()}_{loss_grad.item()}"}

    def sobel_conv(self, x):
        """
        参考CDDFuse实现
        :param x:
        :return:
        """
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_y = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(device)
        weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(device)

        sobel_x = F.conv2d(x, weight_x, padding=1)
        sobel_y = F.conv2d(x, weight_y, padding=1)
        return torch.abs(sobel_x) + torch.abs(sobel_y)


if __name__ == '__main__':
    ir_img = torch.ones(1, 1, 224, 224)
    vis_img = torch.ones(1, 1, 224, 224)
    fus_img = torch.ones(1, 1, 224, 224)
    unreg_fus_img = torch.randn(10, 1, 224, 224)
    print(ir_img.shape, vis_img.shape, fus_img.shape)
    test_feature = [torch.randn(10, 1, 224, 224) for i in range(9)]

    loss = FusionLoss(*test_feature, fus_img, ir_img, vis_img, registration=True).get_loss().to(device)
    print(loss)
