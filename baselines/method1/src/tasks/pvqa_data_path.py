# coding=utf-8

import base64
import json
import os
import pandas as pd
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

import re
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

train_list = ['train_0000', 'train_0001', 'train_0007', 'train_0008', 'train_0009', 'train_0010', 'train_0012', 'train_0013', 'train_0016', 'train_0017', 'train_0020', 'train_0024', 'train_0025', 'train_0027', 'train_0028', 'train_0030', 'train_0032', 'train_0033', 'train_0035', 'train_0037', 'train_0038', 'train_0040', 'train_0041', 'train_0047', 'train_0050', 'train_0052', 'train_0053', 'train_0055', 'train_0056', 'train_0058', 'train_0060', 'train_0063', 'train_0064', 'train_0066', 'train_0069', 'train_0072', 'train_0075', 'train_0076', 'train_0077', 'train_0080', 'train_0081', 'train_0082', 'train_0083', 'train_0084', 'train_0085', 'train_0091', 'train_0092', 'train_0093', 'train_0095', 'train_0100', 'train_0101', 'train_0103', 'train_0105', 'train_0108', 'train_0109', 'train_0110', 'train_0112', 'train_0113', 'train_0114', 'train_0115', 'train_0118', 'train_0119', 'train_0120', 'train_0121', 'train_0123', 'train_0125', 'train_0126', 'train_0127', 'train_0128', 'train_0133', 'train_0136', 'train_0138', 'train_0139', 'train_0140', 'train_0144', 'train_0145', 'train_0146', 'train_0147', 'train_0149', 'train_0151', 'train_0153', 'train_0154', 'train_0156', 'train_0157', 'train_0161', 'train_0162', 'train_0164', 'train_0166', 'train_0171', 'train_0172', 'train_0173', 'train_0175', 'train_0178', 'train_0179', 'train_0185', 'train_0186', 'train_0187', 'train_0189', 'train_0190', 'train_0192', 'train_0193', 'train_0194', 'train_0195', 'train_0196', 'train_0198', 'train_0199', 'train_0200', 'train_0201', 'train_0205', 'train_0206', 'train_0207', 'train_0208', 'train_0209', 'train_0210', 'train_0211', 'train_0214', 'train_0215', 'train_0216', 'train_0217', 'train_0220', 'train_0223', 'train_0224', 'train_0225', 'train_0227', 'train_0231', 'train_0232', 'train_0233', 'train_0236', 'train_0237', 'train_0239', 'train_0240', 'train_0242', 'train_0245', 'train_0246', 'train_0250', 'train_0251', 'train_0253', 'train_0254', 'train_0255', 'train_0258', 'train_0259', 'train_0261', 'train_0264', 'train_0266', 'train_0268', 'train_0269', 'train_0270', 'train_0271', 'train_0274', 'train_0275', 'train_0277', 'train_0278', 'train_0279', 'train_0280', 'train_0281', 'train_0282', 'train_0287', 'train_0288', 'train_0289', 'train_0290', 'train_0291', 'train_0292', 'train_0293', 'train_0294', 'train_0295', 'train_0300', 'train_0301', 'train_0302', 'train_0303', 'train_0304', 'train_0305', 'train_0306', 'train_0308', 'train_0310', 'train_0312', 'train_0313', 'train_0314', 'train_0316', 'train_0319', 'train_0320', 'train_0323', 'train_0324', 'train_0325', 'train_0326', 'train_0329', 'train_0331', 'train_0332', 'train_0334', 'train_0335', 'train_0336', 'train_0340', 'train_0342', 'train_0343', 'train_0344', 'train_0347', 'train_0351', 'train_0352', 'train_0353', 'train_0354', 'train_0357', 'train_0359', 'train_0360', 'train_0361', 'train_0363', 'train_0365', 'train_0366', 'train_0371', 'train_0374', 'train_0376', 'train_0378', 'train_0379', 'train_0381', 'train_0382', 'train_0383', 'train_0386', 'train_0389', 'train_0392', 'train_0393', 'train_0395', 'train_0396', 'train_0397', 'train_0399', 'train_0401', 'train_0405', 'train_0409', 'train_0410', 'train_0411', 'train_0412', 'train_0413', 'train_0414', 'train_0415', 'train_0416', 'train_0418', 'train_0419', 'train_0420', 'train_0422', 'train_0424', 'train_0425', 'train_0426', 'train_0429', 'train_0430', 'train_0432', 'train_0433', 'train_0434', 'train_0435', 'train_0436', 'train_0438', 'train_0439', 'train_0440', 'train_0443', 'train_0445', 'train_0449', 'train_0450', 'train_0452', 'train_0454', 'train_0456', 'train_0457', 'train_0458', 'train_0459', 'train_0460', 'train_0461', 'train_0464', 'train_0465', 'train_0468', 'train_0470', 'train_0471', 'train_0472', 'train_0473', 'train_0474', 'train_0475', 'train_0476', 'train_0477', 'train_0478', 'train_0479', 'train_0481', 'train_0484', 'train_0487', 'train_0490', 'train_0491', 'train_0492', 'train_0493', 'train_0496', 'train_0497', 'train_0499', 'train_0500', 'train_0501', 'train_0503', 'train_0505', 'train_0507', 'train_0509', 'train_0511', 'train_0512', 'train_0513', 'train_0514', 'train_0517', 'train_0520', 'train_0522', 'train_0523', 'train_0524', 'train_0525', 'train_0529', 'train_0530', 'train_0532', 'train_0538', 'train_0540', 'train_0543', 'train_0544', 'train_0545', 'train_0546', 'train_0549', 'train_0550', 'train_0551', 'train_0552', 'train_0553', 'train_0559', 'train_0561', 'train_0562', 'train_0563', 'train_0566', 'train_0567', 'train_0569', 'train_0570', 'train_0572', 'train_0575', 'train_0577', 'train_0578', 'train_0579', 'train_0583', 'train_0585', 'train_0588', 'train_0589', 'train_0590', 'train_0591', 'train_0593', 'train_0595', 'train_0605', 'train_0607', 'train_0615', 'train_0616', 'train_0617', 'train_0618', 'train_0620', 'train_0621', 'train_0624', 'train_0625', 'train_0626', 'train_0627', 'train_0628', 'train_0631', 'train_0633', 'train_0634', 'train_0635', 'train_0636', 'train_0640', 'train_0641', 'train_0645', 'train_0646', 'train_0647', 'train_0648', 'train_0649', 'train_0650', 'train_0651', 'train_0654', 'train_0656', 'train_0657', 'train_0658', 'train_0661', 'train_0662', 'train_0663', 'train_0664', 'train_0667', 'train_0669', 'train_0673', 'train_0676', 'train_0678', 'train_0679', 'train_0683', 'train_0684', 'train_0686', 'train_0688', 'train_0689', 'train_0690', 'train_0691', 'train_0693', 'train_0695', 'train_0697', 'train_0698', 'train_0704', 'train_0705', 'train_0706', 'train_0709', 'train_0710', 'train_0711', 'train_0712', 'train_0713', 'train_0715', 'train_0718', 'train_0719', 'train_0720', 'train_0721', 'train_0722', 'train_0724', 'train_0725', 'train_0730', 'train_0731', 'train_0732', 'train_0733', 'train_0740', 'train_0741', 'train_0743', 'train_0744', 'train_0746', 'train_0747', 'train_0748', 'train_0753', 'train_0754', 'train_0757', 'train_0759', 'train_0763', 'train_0764', 'train_0768', 'train_0769', 'train_0772', 'train_0775', 'train_0777', 'train_0779', 'train_0783', 'train_0784', 'train_0786', 'train_0787', 'train_0788', 'train_0789', 'train_0792', 'train_0795', 'train_0797', 'train_0798', 'train_0799', 'train_0801', 'train_0807', 'train_0808', 'train_0810', 'train_0812', 'train_0813', 'train_0814', 'train_0817', 'train_0820', 'train_0821', 'train_0822', 'train_0823', 'train_0824', 'train_0825', 'train_0826', 'train_0827', 'train_0828', 'train_0829', 'train_0831', 'train_0835', 'train_0836', 'train_0837', 'train_0840', 'train_0842', 'train_0844', 'train_0845', 'train_0847', 'train_0849', 'train_0850', 'train_0851', 'train_0852', 'train_0854', 'train_0855', 'train_0856', 'train_0857', 'train_0858', 'train_0860', 'train_0861', 'train_0862', 'train_0863', 'train_0865', 'train_0868', 'train_0869', 'train_0870', 'train_0871', 'train_0875', 'train_0876', 'train_0878', 'train_0879', 'train_0881', 'train_0882', 'train_0885', 'train_0887', 'train_0889', 'train_0890', 'train_0891', 'train_0894', 'train_0895', 'train_0897', 'train_0898', 'train_0899', 'train_0900', 'train_0901', 'train_0906', 'train_0908', 'train_0911', 'train_0912', 'train_0913', 'train_0915', 'train_0916', 'train_0918', 'train_0925', 'train_0926', 'train_0934', 'train_0935', 'train_0936', 'train_0939', 'train_0941', 'train_0942', 'train_0946', 'train_0947', 'train_0948', 'train_0949', 'train_0950', 'train_0951', 'train_0952', 'train_0955', 'train_0958', 'train_0959', 'train_0960', 'train_0962', 'train_0963', 'train_0966', 'train_0968', 'train_0969', 'train_0971', 'train_0972', 'train_0973', 'train_0974', 'train_0975', 'train_0976', 'train_0979', 'train_0980', 'train_0983', 'train_0984', 'train_0985', 'train_0987', 'train_0990', 'train_0992', 'train_0993', 'train_0994', 'train_0997', 'train_0998', 'train_1000', 'train_1001', 'train_1008', 'train_1009', 'train_1011', 'train_1012', 'train_1013', 'train_1014', 'train_1015', 'train_1016', 'train_1017', 'train_1018', 'train_1019', 'train_1020', 'train_1021', 'train_1022', 'train_1023', 'train_1024', 'train_1025', 'train_1026', 'train_1027', 'train_1028', 'train_1029', 'train_1030', 'train_1031', 'train_1032', 'train_1033', 'train_1034', 'train_1035', 'train_1036', 'train_1037', 'train_1038', 'train_1039', 'train_1040', 'train_1041', 'train_1042', 'train_1043', 'train_1044', 'train_1045', 'train_1046', 'train_1047', 'train_1048', 'train_1049', 'train_1050', 'train_1051', 'train_1052', 'train_1053', 'train_1054', 'train_1055', 'train_1056', 'train_1057', 'train_1058', 'train_1059', 'train_1060', 'train_1061', 'train_1062', 'train_1063', 'train_1064', 'train_1065', 'train_1066', 'train_1067', 'train_1068', 'train_1069', 'train_1070', 'train_1071', 'train_1072', 'train_1073', 'train_1074', 'train_1075', 'train_1076', 'train_1077', 'train_1078', 'train_1079', 'train_1080', 'train_1081', 'train_1082', 'train_1083', 'train_1084', 'train_1087', 'train_1088', 'train_1089', 'train_1090', 'train_1091', 'train_1092', 'train_1093', 'train_1094', 'train_1095', 'train_1096', 'train_1097', 'train_1098', 'train_1099', 'train_1100', 'train_1101', 'train_1102', 'train_1103', 'train_1104', 'train_1105', 'train_1106', 'train_1107', 'train_1108', 'train_1109', 'train_1110', 'train_1111', 'train_1112', 'train_1113', 'train_1114', 'train_1115', 'train_1116', 'train_1117', 'train_1118', 'train_1119', 'train_1120', 'train_1121', 'train_1122', 'train_1123', 'train_1124', 'train_1125', 'train_1126', 'train_1127', 'train_1128', 'train_1129', 'train_1130', 'train_1131', 'train_1132', 'train_1133', 'train_1134', 'train_1135', 'train_1136', 'train_1137', 'train_1138', 'train_1139', 'train_1140', 'train_1141', 'train_1142', 'train_1143', 'train_1144', 'train_1145', 'train_1146', 'train_1147', 'train_1148', 'train_1149', 'train_1150', 'train_1151', 'train_1152', 'train_1153', 'train_1154', 'train_1155', 'train_1156', 'train_1157', 'train_1158', 'train_1159', 'train_1160', 'train_1161', 'train_1162', 'train_1163', 'train_1164', 'train_1165', 'train_1166', 'train_1167', 'train_1168', 'train_1169',
 'train_1170', 'train_1171', 'train_1172', 'train_1173', 'train_1174', 'train_1175', 'train_1176', 'train_1177', 'train_1178', 'train_1179', 'train_1180', 'train_1181', 'train_1182', 'train_1183', 'train_1184', 'train_1185', 'train_1186', 'train_1187', 'train_1188', 'train_1189', 'train_1190', 'train_1191', 'train_1192', 'train_1193', 'train_1194', 'train_1195', 'train_1196', 'train_1197', 'train_1198', 'train_1199', 'train_1200', 'train_1201', 'train_1202', 'train_1203', 'train_1204', 'train_1205', 'train_1206', 'train_1207', 'train_1208', 'train_1266', 'train_1267', 'train_1272', 'train_1273', 'train_1274', 'train_1275', 'train_1276', 'train_1277', 'train_1278', 'train_1279', 'train_1280', 'train_1281', 'train_1282', 'train_1283', 'train_1284', 'train_1285', 'train_1286', 'train_1287', 'train_1288', 'train_1289', 'train_1290', 'train_1291', 'train_1292', 'train_1293', 'train_1294', 'train_1295', 'train_1296', 'train_1297', 'train_1298', 'train_1299', 'train_1300', 'train_1301', 'train_1302', 'train_1303', 'train_1304', 'train_1305', 'train_1306', 'train_1307', 'train_1308', 'train_1309', 'train_1310', 'train_1311', 'train_1312', 'train_1313', 'train_1314', 'train_1315', 'train_1316', 'train_1317', 'train_1318', 'train_1319', 'train_1320', 'train_1321', 'train_1322', 'train_1323', 'train_1324', 'train_1325', 'train_1326', 'train_1327', 'train_1328', 'train_1329', 'train_1330', 'train_1331', 'train_1332', 'train_1333', 'train_1500', 'train_1501', 'train_1502', 'train_1503', 'train_1504', 'train_1505', 'train_1506', 'train_1507', 'train_1508', 'train_1509', 'train_1510', 'train_1511', 'train_1512', 'train_1513', 'train_1514', 'train_1515', 'train_1516', 'train_1517', 'train_1518', 'train_1519', 'train_1520', 'train_1521', 'train_1522', 'train_1523', 'train_1524', 'train_1525', 'train_1526', 'train_1527', 'train_1528', 'train_1529', 'train_1530', 'train_1556', 'train_1557', 'train_1558', 'train_1559', 'train_1560', 'train_1561', 'train_1562', 'train_1563', 'train_1564', 'train_1565', 'train_1566', 'train_1567', 'train_1568', 'train_1569', 'train_1570', 'train_1571', 'train_1572', 'train_1573', 'train_1647', 'train_1648', 'train_1649', 'train_1650', 'train_1651', 'train_1652', 'train_1653', 'train_1654', 'train_1655', 'train_1656', 'train_1657', 'train_1658', 'train_1659', 'train_1660', 'train_1661', 'train_1662', 'train_1663', 'train_1664', 'train_1665', 'train_1666', 'train_1667', 'train_1668', 'train_1669', 'train_1670', 'train_1671', 'train_1672', 'train_1673', 'train_1674', 'train_1675', 'train_1676', 'train_1677', 'train_1678', 'train_1679', 'train_1680', 'train_1681', 'train_1682', 'train_1683', 'train_1684', 'train_1685', 'train_1686', 'train_1687', 'train_1688', 'train_1689', 'train_1690', 'train_1691', 'train_1692', 'train_1693', 'train_1694', 'train_1695', 'train_1696', 'train_1697', 'train_1698', 'train_1768', 'train_1769', 'train_1770', 'train_1771', 'train_1772', 'train_1773', 'train_1774', 'train_1777', 'train_1778', 'train_1779', 'train_1780', 'train_1781', 'train_1782', 'train_1783', 'train_1784', 'train_1785', 'train_1786', 'train_1787', 'train_1788', 'train_1789', 'train_1790', 'train_1791', 'train_1792', 'train_1793', 'train_1794', 'train_1795', 'train_1796', 'train_1797', 'train_1798', 'train_1799', 'train_1800', 'train_1801', 'train_1802', 'train_1803', 'train_1804', 'train_1805', 'train_1806', 'train_1807', 'train_1808', 'train_1809', 'train_1810', 'train_1811', 'train_1812', 'train_1813', 'train_1814', 'train_1815', 'train_1816', 'train_1817', 'train_1818', 'train_1819', 'train_1820', 'train_1821', 'train_1822', 'train_1823', 'train_1824', 'train_1825', 'train_1826', 'train_1827', 'train_1828', 'train_1829', 'train_1830', 'train_1831', 'train_1832', 'train_1833', 'train_1834', 'train_1835', 'train_1836', 'train_1837', 'train_1838', 'train_1839', 'train_1840', 'train_1904', 'train_1905', 'train_1906', 'train_1907', 'train_1908', 'train_1909', 'train_1910', 'train_1911', 'train_1912', 'train_1913', 'train_1914', 'train_1915', 'train_1916', 'train_1917', 'train_1918', 'train_1919', 'train_1920', 'train_1969', 'train_1970', 'train_1971', 'train_1972', 'train_1973', 'train_1974', 'train_1975', 'train_1976', 'train_1977', 'train_1978', 'train_1979', 'train_1980', 'train_1981', 'train_1982', 'train_1983', 'train_1984', 'train_1985', 'train_1986', 'train_1987', 'train_1988', 'train_1989', 'train_1990', 'train_1991', 'train_1992', 'train_1993', 'train_1994', 'train_1995', 'train_1996', 'train_1997', 'train_1998', 'train_1999', 'train_2000', 'train_2001', 'train_2002', 'train_2003', 'train_2004', 'train_2005', 'train_2006', 'train_2007', 'train_2008', 'train_2009', 'train_2010', 'train_2011', 'train_2012', 'train_2013', 'train_2014', 'train_2015', 'train_2016', 'train_2017', 'train_2018', 'train_2019', 'train_2020', 'train_2021', 'train_2022', 'train_2023', 'train_2024', 'train_2025', 'train_2026', 'train_2027', 'train_2028', 'train_2029', 'train_2115', 'train_2116', 'train_2117', 'train_2118', 'train_2119', 'train_2120', 'train_2121', 'train_2122', 'train_2123', 'train_2124', 'train_2125', 'train_2126', 'train_2127', 'train_2128', 'train_2129', 'train_2130', 'train_2131', 'train_2132', 'train_2133', 'train_2134', 'train_2135', 'train_2136', 'train_2137', 'train_2138', 'train_2139', 'train_2140', 'train_2141', 'train_2142', 'train_2143', 'train_2144', 'train_2145', 'train_2146', 'train_2147', 'train_2148', 'train_2149', 'train_2150', 'train_2151', 'train_2152', 'train_2153', 'train_2154', 'train_2155', 'train_2215', 'train_2216', 'train_2217', 'train_2218', 'train_2219', 'train_2220', 'train_2221', 'train_2222', 'train_2223', 'train_2224', 'train_2225', 'train_2226', 'train_2227', 'train_2228', 'train_2229', 'train_2230', 'train_2231', 'train_2232', 'train_2233', 'train_2234', 'train_2235', 'train_2236', 'train_2237', 'train_2238', 'train_2239', 'train_2240', 'train_2241', 'train_2242', 'train_2243', 'train_2244', 'train_2245', 'train_2246', 'train_2247', 'train_2248', 'train_2249', 'train_2250', 'train_2251', 'train_2252', 'train_2253', 'train_2254', 'train_2255', 'train_2256', 'train_2257', 'train_2258', 'train_2259', 'train_2260', 'train_2261', 'train_2262', 'train_2263', 'train_2264', 'train_2265', 'train_2266', 'train_2267', 'train_2268', 'train_2269', 'train_2270', 'train_2271', 'train_2272', 'train_2273', 'train_2274', 'train_2275', 'train_2276', 'train_2277', 'train_2278', 'train_2279', 'train_2280', 'train_2281', 'train_2282', 'train_2283', 'train_2284', 'train_2285', 'train_2286', 'train_2287', 'train_2288', 'train_2289', 'train_2290', 'train_2291', 'train_2292', 'train_2293', 'train_2294', 'train_2295', 'train_2296', 'train_2297', 'train_2298', 'train_2299', 'train_2300', 'train_2301', 'train_2302', 'train_2303', 'train_2304', 'train_2305', 'train_2306', 'train_2307', 'train_2308', 'train_2309', 'train_2310', 'train_2311', 'train_2312', 'train_2313', 'train_2314', 'train_2315', 'train_2316', 'train_2317', 'train_2318', 'train_2319', 'train_2320', 'train_2321', 'train_2322', 'train_2323', 'train_2324', 'train_2325', 'train_2326', 'train_2327', 'train_2328', 'train_2329', 'train_2330', 'train_2331', 'train_2332', 'train_2333', 'train_2334', 'train_2335', 'train_2336', 'train_2337', 'train_2338', 'train_2339', 'train_2340', 'train_2341', 'train_2342', 'train_2343', 'train_2344', 'train_2345', 'train_2346', 'train_2347', 'train_2348', 'train_2349', 'train_2350', 'train_2351', 'train_2352', 'train_2353', 'train_2354', 'train_2355', 'train_2356', 'train_2357', 'train_2358', 'train_2532', 'train_2533', 'train_2534', 'train_2535', 'train_2536', 'train_2537', 'train_2538', 'train_2539', 'train_2540', 'train_2541', 'train_2542', 'train_2543', 'train_2544', 'train_2545', 'train_2546', 'train_2547', 'train_2548', 'train_2549', 'train_2550', 'train_2551', 'train_2552', 'train_2553', 'train_2554', 'train_2555', 'train_2556', 'train_2557', 'train_2558', 'train_2559', 'train_2560', 'train_2561', 'train_2562', 'train_2563', 'train_2564', 'train_2565', 'train_2566', 'train_2567', 'train_2568', 'train_2569', 'train_2570', 'train_2571', 'train_2572', 'train_2573', 'train_2574', 'train_2575', 'train_2576', 'train_2577', 'train_2578', 'train_2579', 'train_2580', 'train_2581', 'train_2582', 'train_2583', 'train_2741', 'train_2742', 'train_2743', 'train_2744', 'train_2745', 'train_2746', 'train_2747', 'train_2748', 'train_2749', 'train_2750', 'train_2751', 'train_2752', 'train_2753', 'train_2754', 'train_2755', 'train_2756', 'train_2757', 'train_2758', 'train_2759', 'train_2760', 'train_2761', 'train_2762', 'train_2763', 'train_2764', 'train_2765', 'train_2766', 'train_2767', 'train_2768', 'train_2769', 'train_2770', 'train_2771', 'train_2772', 'train_2773', 'train_2774', 'train_2775', 'train_2776', 'train_2777', 'train_2778', 'train_2779', 'train_2780', 'train_2781', 'train_2782', 'train_2783', 'train_2784', 'train_2785', 'train_2786', 'train_2787', 'train_2788', 'train_2789', 'train_2790', 'train_2791', 'train_2792', 'train_2793', 'train_2794', 'train_2795', 'train_2796', 'train_2909', 'train_2910', 'train_2911', 'train_2912', 'train_2913', 'train_2914', 'train_2915', 'train_2916', 'train_2917', 'train_2918', 'train_2919', 'train_2920', 'train_2921', 'train_2922', 'train_2923', 'train_2924', 'train_2925', 'train_2926', 'train_2927', 'train_2928', 'train_2929', 'train_2930', 'train_2931', 'train_2932', 'train_2933', 'train_2934', 'train_2935', 'train_2936', 'train_2937', 'train_2938', 'train_2939', 'train_2940', 'train_2945', 'train_2946', 'train_2947', 'train_2948', 'train_2949', 'train_2950', 'train_2951', 'train_2952', 'train_2953', 'train_2954', 'train_2955', 'train_2956', 'train_2957', 'train_2958', 'train_2959', 'train_2960', 'train_2961', 'train_2962', 'train_2963', 'train_2964', 'train_2965', 'train_2966', 'train_2967', 'train_2968', 'train_2969', 'train_2970', 'train_2971', 'train_2972', 'train_2973', 'train_2974', 'train_2975', 'train_2976', 'train_2977', 'train_2978', 'train_2979', 
 'train_2980', 'train_2981', 'train_2982', 'train_2983', 'train_2984', 'train_2985', 'train_2986', 'train_2987', 'train_2988', 'train_2989', 'train_2990', 'train_2991', 'train_2992', 'train_2993', 'train_2994', 'train_2995', 'train_2996', 'train_2997', 'train_2998', 'train_2999', 'train_3000', 'train_3001', 'train_3002', 'train_3003', 'train_3004', 'train_3005', 'train_3006', 'train_3007', 'train_3008', 'train_3009', 'train_3010', 'train_3011', 'train_3012', 'train_3013', 'train_3014', 'train_3015', 'train_3016', 'train_3018', 'train_3019']

val_list = ['val_0000', 'val_0001', 'val_0003', 'val_0004', 'val_0006', 'val_0010', 'val_0011', 'val_0013', 'val_0014', 'val_0015', 'val_0016', 'val_0017', 'val_0018', 'val_0019', 'val_0021', 'val_0029', 'val_0030', 'val_0031', 'val_0033', 'val_0036', 'val_0037', 'val_0038', 'val_0040', 'val_0041', 'val_0042', 'val_0048', 'val_0049', 'val_0052', 'val_0053', 'val_0054', 'val_0055', 'val_0056', 'val_0057', 'val_0058', 'val_0061', 'val_0065', 'val_0067', 'val_0068', 'val_0069', 'val_0071', 'val_0073', 'val_0074', 'val_0075', 'val_0076', 'val_0077', 'val_0080', 'val_0081', 'val_0082', 'val_0083', 'val_0084', 'val_0085', 'val_0090', 'val_0091', 'val_0092', 'val_0093', 'val_0094', 'val_0095', 'val_0096', 'val_0097', 'val_0098', 'val_0099', 'val_0100', 'val_0101', 'val_0103', 'val_0104', 'val_0105', 'val_0106', 'val_0107', 'val_0108', 'val_0109', 'val_0110', 'val_0112', 'val_0113', 'val_0114', 'val_0115', 'val_0116', 'val_0119', 'val_0120', 'val_0121', 'val_0122', 'val_0123', 'val_0126', 'val_0127', 'val_0128', 'val_0129', 'val_0130', 'val_0132', 'val_0133', 'val_0140', 'val_0141', 'val_0142', 'val_0143', 'val_0144', 'val_0145', 'val_0146', 'val_0147', 'val_0148', 'val_0149', 'val_0152', 'val_0153', 'val_0154', 'val_0155', 'val_0156', 'val_0157', 'val_0158', 'val_0160', 'val_0162', 'val_0165', 'val_0167', 'val_0169', 'val_0170', 'val_0171', 'val_0172', 'val_0173', 'val_0174', 'val_0177', 'val_0178', 'val_0186', 'val_0187', 'val_0188', 'val_0189', 'val_0190', 'val_0191', 'val_0193', 'val_0194', 'val_0196', 'val_0197', 'val_0198', 'val_0200', 'val_0202', 'val_0203', 'val_0204', 'val_0205', 'val_0207', 'val_0209', 'val_0211', 'val_0212', 'val_0213', 'val_0215', 'val_0220', 'val_0221', 'val_0223', 'val_0224', 'val_0226', 'val_0228', 'val_0229', 'val_0230', 'val_0233', 'val_0236', 'val_0238', 'val_0243', 'val_0244', 'val_0245', 'val_0246', 'val_0249', 'val_0253', 'val_0255', 'val_0256', 'val_0257', 'val_0258', 'val_0259', 'val_0264', 'val_0265', 'val_0268', 'val_0269', 'val_0271', 'val_0272', 'val_0273', 'val_0274', 'val_0276', 'val_0279', 'val_0282', 'val_0283', 'val_0284', 'val_0285', 'val_0286', 'val_0287', 'val_0290', 'val_0292', 'val_0293', 'val_0294', 'val_0297', 'val_0298', 'val_0299', 'val_0301', 'val_0302', 'val_0303', 'val_0306', 'val_0307', 'val_0309', 'val_0310', 'val_0311', 'val_0313', 'val_0314', 'val_0315', 'val_0319', 'val_0322', 'val_0323', 'val_0325', 'val_0326', 'val_0329', 'val_0330', 'val_0333', 'val_0334', 'val_0337', 'val_0338', 'val_0340', 'val_0341', 'val_0342', 'val_0343', 'val_0344', 'val_0345', 'val_0346', 'val_0347', 'val_0348', 'val_0349', 'val_0350', 'val_0351', 'val_0352', 'val_0353', 'val_0354', 'val_0355', 'val_0356', 'val_0357', 'val_0358', 'val_0359', 'val_0360', 'val_0361', 'val_0362', 'val_0363', 'val_0364', 'val_0365', 'val_0366', 'val_0367', 'val_0368', 'val_0369', 'val_0370', 'val_0371', 'val_0372', 'val_0373', 'val_0374', 'val_0375', 'val_0376', 'val_0377', 'val_0378', 'val_0379', 'val_0380', 'val_0381', 'val_0382', 'val_0383', 'val_0384', 'val_0385', 'val_0386', 'val_0387', 'val_0388', 'val_0389', 'val_0390', 'val_0391', 'val_0392', 'val_0393', 'val_0394', 'val_0395', 'val_0396', 'val_0397', 'val_0398', 'val_0399', 'val_0422', 'val_0423', 'val_0424', 'val_0425', 'val_0426', 'val_0427', 'val_0428', 'val_0429', 'val_0430', 'val_0431', 'val_0432', 'val_0433', 'val_0434', 'val_0435', 'val_0436', 'val_0437', 'val_0438', 'val_0439', 'val_0440', 'val_0441', 'val_0479', 'val_0480', 'val_0499', 'val_0500', 'val_0501', 'val_0502', 'val_0503', 'val_0504', 'val_0505', 'val_0506', 'val_0507', 'val_0508', 'val_0509', 'val_0510', 'val_0511', 'val_0512', 'val_0526', 'val_0527', 'val_0528', 'val_0529', 'val_0530', 'val_0531', 'val_0532', 'val_0547', 'val_0551', 'val_0552', 'val_0553', 'val_0554', 'val_0555', 'val_0556', 'val_0558', 'val_0559', 'val_0560', 'val_0561', 'val_0562', 'val_0563', 'val_0591', 'val_0592', 'val_0593', 'val_0594', 'val_0595', 'val_0596', 'val_0597', 'val_0598', 'val_0599', 'val_0600', 'val_0601', 'val_0602', 'val_0603', 'val_0604', 'val_0605', 'val_0606', 'val_0607', 'val_0608', 'val_0609', 'val_0610', 'val_0625', 'val_0626', 'val_0627', 'val_0628', 'val_0644', 'val_0645', 'val_0646', 'val_0647', 'val_0648', 'val_0649', 'val_0650', 'val_0651', 'val_0652', 'val_0653', 'val_0654', 'val_0655', 'val_0656', 'val_0657', 'val_0696', 'val_0697', 'val_0698', 'val_0699', 'val_0700', 'val_0701', 'val_0702', 'val_0703', 'val_0704', 'val_0705', 'val_0706', 'val_0707', 'val_0708', 'val_0709', 'val_0710', 'val_0736', 'val_0737', 'val_0738', 'val_0739', 'val_0740', 'val_0741', 'val_0742', 'val_0743', 'val_0744', 'val_0745', 'val_0746', 'val_0747', 'val_0748', 'val_0749', 'val_0750', 'val_0751', 'val_0752', 'val_0753', 'val_0754', 'val_0755', 'val_0756', 'val_0757', 'val_0758', 'val_0759', 'val_0760', 'val_0761', 'val_0762', 'val_0763', 'val_0764', 'val_0765', 'val_0766', 'val_0767', 'val_0768', 'val_0769', 'val_0770', 'val_0818', 'val_0819', 'val_0820', 'val_0821', 'val_0822', 'val_0823', 'val_0824', 'val_0825', 'val_0826', 'val_0827', 'val_0828', 'val_0829', 'val_0830', 'val_0831', 'val_0832', 'val_0833', 'val_0834', 'val_0835', 'val_0836', 'val_0837', 'val_0838', 'val_0839', 'val_0840', 'val_0841', 'val_0842', 'val_0843', 'val_0844', 'val_0845', 'val_0846', 'val_0847', 'val_0848', 'val_0905', 'val_0906', 'val_0907', 'val_0908', 'val_0909', 'val_0910', 'val_0911', 'val_0912', 'val_0913', 'val_0914', 'val_0915', 'val_0916', 'val_0917', 'val_0918', 'val_0919', 'val_0920', 'val_0921', 'val_0964', 'val_0965', 'val_0966', 'val_0967', 'val_0968', 'val_0969', 'val_0970', 'val_0971', 'val_0972', 'val_0973', 'val_0974', 'val_0975', 'val_0976', 'val_0977', 'val_0978', 'val_0979', 'val_0980', 'val_0981', 'val_0982', 'val_0983', 'val_0984', 'val_0985', 'val_0986', 'val_0987', 'val_0988', 'val_0989', 'val_0991']

test_list = ['test_0001', 'test_0005', 'test_0006', 'test_0009', 'test_0011', 'test_0012', 'test_0013', 'test_0017', 'test_0018', 'test_0019', 'test_0021', 'test_0022', 'test_0023', 'test_0024', 'test_0025', 'test_0026', 'test_0031', 'test_0032', 'test_0035', 'test_0036', 'test_0037', 'test_0038', 'test_0040', 'test_0041', 'test_0042', 'test_0043', 'test_0044', 'test_0045', 'test_0048', 'test_0050', 'test_0055', 'test_0056', 'test_0057', 'test_0060', 'test_0061', 'test_0062', 'test_0063', 'test_0065', 'test_0067', 'test_0068', 'test_0069', 'test_0070', 'test_0071', 'test_0072', 'test_0073', 'test_0076', 'test_0078', 'test_0080', 'test_0081', 'test_0084', 'test_0085', 'test_0087', 'test_0088', 'test_0089', 'test_0093', 'test_0098', 'test_0102', 'test_0104', 'test_0108', 'test_0110', 'test_0111', 'test_0112', 'test_0113', 'test_0115', 'test_0116', 'test_0120', 'test_0122', 'test_0123', 'test_0124', 'test_0128', 'test_0129', 'test_0130', 'test_0131', 'test_0133', 'test_0137', 'test_0138', 'test_0140', 'test_0142', 'test_0143', 'test_0144', 'test_0148', 'test_0151', 'test_0152', 'test_0153', 'test_0154', 'test_0157', 'test_0159', 'test_0162', 'test_0163', 'test_0165', 'test_0166', 'test_0168', 'test_0169', 'test_0173', 'test_0177', 'test_0180', 'test_0183', 'test_0184', 'test_0187', 'test_0188', 'test_0189', 'test_0190', 'test_0191', 'test_0193', 'test_0194', 'test_0195', 'test_0198', 'test_0199', 'test_0206', 'test_0209', 'test_0210', 'test_0211', 'test_0212', 'test_0216', 'test_0217', 'test_0218', 'test_0219', 'test_0220', 'test_0225', 'test_0226', 'test_0228', 'test_0232', 'test_0233', 'test_0234', 'test_0235', 'test_0236', 'test_0237', 'test_0238', 'test_0239', 'test_0241', 'test_0245', 'test_0246', 'test_0247', 'test_0249', 'test_0252', 'test_0253', 'test_0254', 'test_0255', 'test_0256', 'test_0259', 'test_0264', 'test_0265', 'test_0272', 'test_0274', 'test_0275', 'test_0278', 'test_0279', 'test_0280', 'test_0282', 'test_0284', 'test_0285', 'test_0288', 'test_0291', 'test_0294', 'test_0295', 'test_0297', 'test_0298', 'test_0299', 'test_0304', 'test_0310', 'test_0312', 'test_0313', 'test_0315', 'test_0317', 'test_0318', 'test_0319', 'test_0320', 'test_0321', 'test_0322', 'test_0323', 'test_0324', 'test_0325', 'test_0326', 'test_0327', 'test_0328', 'test_0329', 'test_0330', 'test_0331', 'test_0332', 'test_0333', 'test_0334', 'test_0335', 'test_0336', 'test_0337', 'test_0338', 'test_0339', 'test_0340', 'test_0341', 'test_0342', 'test_0343', 'test_0344', 'test_0345', 'test_0346', 'test_0347', 'test_0348', 'test_0349', 'test_0350', 'test_0351', 'test_0352', 'test_0353', 'test_0354', 'test_0355', 'test_0356', 'test_0357', 'test_0358', 'test_0359', 'test_0360', 'test_0361', 'test_0362', 'test_0363', 'test_0364', 'test_0365', 'test_0366', 'test_0367', 'test_0368', 'test_0369', 'test_0370', 'test_0371', 'test_0372', 'test_0373', 'test_0374', 'test_0393', 'test_0394', 'test_0395', 'test_0396', 'test_0397', 'test_0398', 'test_0399', 'test_0400', 'test_0401', 'test_0402', 'test_0403', 'test_0404', 'test_0405', 'test_0406', 'test_0407', 'test_0408', 'test_0409', 'test_0410', 'test_0436', 'test_0462', 'test_0463', 'test_0464', 'test_0465', 'test_0466', 'test_0467', 'test_0468', 'test_0469', 'test_0470', 'test_0471', 'test_0472', 'test_0473', 'test_0474', 'test_0484', 'test_0485', 'test_0486', 'test_0487', 'test_0488', 'test_0489', 'test_0490', 'test_0491', 'test_0492', 'test_0493', 'test_0494', 'test_0495', 'test_0513', 'test_0514', 'test_0515', 'test_0516', 'test_0517', 'test_0518', 'test_0519', 'test_0520', 'test_0521', 'test_0522', 'test_0523', 'test_0524', 'test_0525', 'test_0526', 'test_0527', 'test_0528', 'test_0529', 'test_0530', 'test_0531', 'test_0532', 'test_0533', 'test_0534', 'test_0535', 'test_0536', 'test_0537', 'test_0561', 'test_0562', 'test_0563', 'test_0564', 'test_0565', 'test_0566', 'test_0567', 'test_0568', 'test_0569', 'test_0570', 'test_0571', 'test_0572', 'test_0573', 'test_0574', 'test_0575', 'test_0576', 'test_0577', 'test_0578', 'test_0579', 'test_0580', 'test_0581', 'test_0582', 'test_0606', 'test_0607', 'test_0608', 'test_0609', 'test_0610', 'test_0631', 'test_0632', 'test_0633', 'test_0634', 'test_0635', 'test_0636', 'test_0637', 'test_0638', 'test_0639', 'test_0640', 'test_0641', 'test_0642', 'test_0643', 'test_0644', 'test_0645', 'test_0646', 'test_0647', 'test_0648', 'test_0649', 'test_0650', 'test_0651', 'test_0652', 'test_0653', 'test_0654', 'test_0655', 'test_0692', 'test_0693', 'test_0694', 'test_0695', 'test_0696', 'test_0697', 'test_0698', 'test_0699', 'test_0700', 'test_0701', 'test_0702', 'test_0703', 'test_0704', 'test_0705', 'test_0706', 'test_0707', 'test_0724', 'test_0725', 'test_0726', 'test_0727', 'test_0728', 'test_0729', 'test_0730', 'test_0731', 'test_0732', 'test_0733', 'test_0734', 'test_0735', 'test_0736', 'test_0737', 'test_0738', 'test_0739', 'test_0740', 'test_0741', 'test_0742', 'test_0743', 'test_0744', 'test_0745', 'test_0746', 'test_0747', 'test_0748', 'test_0749', 'test_0750', 'test_0751', 'test_0752', 'test_0753', 'test_0754', 'test_0755', 'test_0756', 'test_0757', 'test_0758', 'test_0759', 'test_0760', 'test_0761', 'test_0762', 'test_0763', 'test_0764', 'test_0765', 'test_0766', 'test_0767', 'test_0768', 'test_0805', 'test_0812', 'test_0813', 'test_0814', 'test_0815', 'test_0816', 'test_0817', 'test_0818', 'test_0819', 'test_0820', 'test_0821', 'test_0822', 'test_0823', 'test_0824', 'test_0825', 'test_0826', 'test_0827', 'test_0828', 'test_0887', 'test_0888', 'test_0889', 'test_0890', 'test_0891', 'test_0892', 'test_0893', 'test_0894', 'test_0895', 'test_0896', 'test_0897', 'test_0898', 'test_0899', 'test_0900', 'test_0901', 'test_0902', 'test_0903', 'test_0950', 'test_0951', 'test_0952', 'test_0953', 'test_0954', 'test_0955', 'test_0956', 'test_0957', 'test_0958', 'test_0959', 'test_0960', 'test_0961', 'test_0962', 'test_0963', 'test_0964', 'test_0965', 'test_0966', 'test_0967', 'test_0968', 'test_0969', 'test_0970', 'test_0971', 'test_0972', 'test_0973', 'test_0974', 'test_0975', 'test_0976', 'test_0977', 'test_0978', 'test_0979', 'test_0980', 'test_0981', 'test_0982', 'test_0983', 'test_0984', 'test_0985', 'test_0986', 'test_0987', 'test_0988', 'test_0989', 'test_0990']

baseUrl = 'drive/MyDrive/PathVQA/'

FIELDNAMES = ['image_id', 'image_w', 'image_h',
              'num_boxes', 'boxes', 'features']


def load_tsv(split: str):
    tsv_file = baseUrl+'data/pvqa/images/%s%s.csv' % (split, args.pvqaimgv)
    df = pd.read_csv(tsv_file, delimiter='\t', names=FIELDNAMES)

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum['img_id'] = '%s_%04d' % (split, df['image_id'][i])
        datum['img_w'] = df['image_w'][i]
        datum['img_h'] = df['image_h'][i]
        datum['num_boxes'] = df['num_boxes'][i]

        boxes = df['boxes'][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum['boxes'] = temp.reshape(datum['num_boxes'], -1)

        features = df['features'][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum['features'] = temp.reshape(datum['num_boxes'], -1)

        data.append(datum)

    return data


class PVQADataset:

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # loading dataset
        self.data = []
        for split in self.splits:
            self.data.extend(pickle.load(
                open(baseUrl+'data/pvqa/qas/%s_vqa.pkl' % split, 'rb')))
        print('Load %d data from splits %s' % (len(self.data), self.name))
        # Convert list to dict for evaluation
        self.id2datum = {datum['question_id']: datum for datum in self.data}

        # Answers
        # self.q2a = pickle.load(open('data/pvqa/qas/q2a.pkl', 'rb'))
        # self.qid2a = pickle.load(open('data/pvqa/qas/qid2a.pkl', 'rb'))
        # self.qid2q = pickle.load(open('data/pvqa/qas/qid2q.pkl', 'rb'))
        self.ans2label = pickle.load(
            open(baseUrl+'data/pvqa/qas/trainval_path_ans2label.pkl', 'rb'))
        self.label2ans = pickle.load(
            open(baseUrl+'data/pvqa/qas/trainval_path_label2ans.pkl', 'rb'))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class PVQATorchDataset(Dataset):
    def __init__(self, dataset: PVQADataset):
        super(PVQATorchDataset, self).__init__()
        self.raw_dataset = dataset

        # loading detection features to img_data
        self.imgid2img = {}
        for split in dataset.splits:
            data = load_tsv(split)

            if split == "train":
                path_list = train_list
            elif split == "val":
                path_list = val_list
            elif split == "test":
                path_list = test_list

            for datum in data:
                if datum['img_id'] in path_list:
                    self.imgid2img[datum['img_id']] = datum

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print('use %d data in torch dataset' % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()

        assert obj_num == len(boxes) == len(feats)

        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target, img_id, img_info
        else:
            return ques_id, feats, boxes, ques, img_id, img_info


question_types = ('where', 'what', 'how', 'how many/how much',
                  'when', 'why', 'who/whose', 'other', 'yes/no')


def get_q_type(q: str):
    q = q.lower()
    if q.startswith('how many') or q.startswith('how much'):
        return 'how many/how much'
    first_w = q.split()[0]
    if first_w in ('who', 'whose'):
        return 'who/whose'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if first_w == q_type:
            return q_type
    if first_w in ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'does', 'do', 'did', 'can', 'could']:
        return 'yes/no'
    if 'whose' in q:
        return 'who/whose'
    if 'how many' in q or 'how much' in q:
        return 'how many/how much'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if q_type in q:
            return q_type
    print(q)
    return 'other'


class PVQAEvaluator:
    def __init__(self, dataset: PVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        qtype_score = {qtype: 0. for qtype in question_types}
        qtype_cnt = {qtype: 0 for qtype in question_types}
        preds = []
        anss = []
        b_scores = []
        b_scores1 = []
        b_scores2 = []
        b_scores3 = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            quest = datum['sent']

            q_type = get_q_type(quest)
            qtype_cnt[q_type] += 1

            hypo = str(ans).lower().split()
            refs = []
            preds.append(self.dataset.ans2label[ans])
            if ans in label:
                score += label[ans]
                qtype_score[q_type] += label[ans]
            ans_flag = True
            for gt_ans in label:
                refs.append(str(gt_ans).lower().split())
                if ans_flag:
                    anss.append(
                        self.dataset.ans2label[gt_ans] if gt_ans in self.dataset.ans2label else -1)
                    ans_flag = False
            b_score = sentence_bleu(references=refs, hypothesis=hypo)
            b_score1 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[1, 0, 0, 0])
            b_score2 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[0, 1, 0, 0])
            b_score3 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[0, 0, 1, 0])

            b_scores.append(b_score)
            b_scores1.append(b_score1)
            b_scores2.append(b_score2)
            b_scores3.append(b_score3)
        b_score_m = np.mean(b_scores)
        b_score_m1 = np.mean(b_scores1)
        b_score_m2 = np.mean(b_scores2)
        b_score_m3 = np.mean(b_scores3)
        info = 'b_score=%.4f\n' % b_score_m
        info += 'b_score1 = %.4f\n' % b_score_m1
        info += 'b_score2 = %.4f\n' % b_score_m2
        info += 'b_score3 = %.4f\n' % b_score_m3

        info += 'f1_score=%.4f\n' % f1_score(anss, preds, average='macro')
        info += 'score = %.4f\n' % (score / len(quesid2ans))
        for q_type in question_types:
            if qtype_cnt[q_type] > 0:
                qtype_score[q_type] /= qtype_cnt[q_type]
        info += 'Overall score: %.4f\n' % (score / len(quesid2ans))
        for q_type in question_types:
            info += 'qtype: %s\t score=%.4f\n' % (q_type, qtype_score[q_type])

        with open(os.path.join(args.output, 'result_by_type.txt'), 'a') as f:
            f.write(info)
        print(info)
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
