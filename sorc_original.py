import numpy as np
import pandas as pd
import h5py
import os
from bs4 import BeautifulSoup

# Write dataframe with statistics in the correct folder structure
def write_dframe(savepath, pp_ms_parameters, stats_eval_dict, stats_sort_dict):
    dframe_dict = dict_to_dframe(stats_eval_dict, stats_sort_dict)
    filepath = os.path.join(savepath, os.path.basename(savepath) +  "_sorc_evaluation.h5")
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    #if os.path.exists(filepath):
    #    raise ValueError("File already exists, please change the savepath!")
    with h5py.File(filepath, "a") as hdf:
        g = hdf.create_group(pp_ms_parameters)
        string_dt = h5py.special_dtype(vlen=str)
        for clustermethod, dframe in dframe_dict.items():
            g.create_dataset("columns", data=np.array(dframe.columns, dtype="S"), dtype=string_dt)
            g.create_dataset("rows", data=np.array(dframe.index, dtype="S"), dtype=string_dt)
            break
        for clustermethod, dframe in dframe_dict.items():
            g.create_dataset(clustermethod, data=dframe.astype(float))


def dict_to_dframe(stats_eval_dict, stats_sort_dict):
    # Array of sorted stats to have consistent dataframes
    sorted_stats = [stats_sort_dict[key][0] for key in sorted(stats_sort_dict.keys())]
    stat_optimization_orientation = [stats_sort_dict[key][1] for key in sorted(stats_sort_dict.keys())]
    # One dataframe for each cluster method
    dframe_dict = {}
    for cl_method, dict_of_measures in stats_eval_dict.items():
        dframe = pd.DataFrame(index=dict_of_measures.keys(), columns=sorted_stats)
        for measure, dict_of_stats in dict_of_measures.items():
            # Get stat values in the correct order and place them into the correct dataframe row
            stat_values = [dict_of_stats[stat] for stat in sorted_stats]
            dframe.loc[measure] = stat_values
        dframe_dict[cl_method] = dframe
    return dframe_dict



def write_sorc(hdfpath, savepath, limit):
    reindex_dict = {
        "noPPnoMS": "no PP -- no MS -- ",
        "noPP1MS": "no PP --  1 MS -- ",
        "noPP2MS": "no PP --  2 MS -- ",
        "PPnoMS": "   PP -- no MS -- ",
        "PP1MS": "   PP --  1 MS -- ",
        "PP2MS": "   PP --  2 MS -- "
    }
    name = os.path.basename(hdfpath).split(".")[0]
    dframe_dict = {}
    with h5py.File(hdfpath, "r") as hdf:
        ppmss = list(hdf.keys())
        for ppms in ppmss:
            columns = hdf[ppms]["columns"]
            rows = hdf[ppms]["rows"]
            clustermethods = list(hdf[ppms].keys())
            clustermethods.remove("rows")
            clustermethods.remove("columns")
            for cm in clustermethods:
                if cm not in dframe_dict.keys():
                    dframe_dict[cm] = pd.DataFrame(columns=columns)
                    
            for cm in clustermethods:
                dframe = dframe_dict[cm]
                h5pyframe = pd.DataFrame(hdf[ppms][cm], index=rows, columns=columns)
                h5pyframe.index = [reindex_dict[ppms] + idx for idx in h5pyframe.index]
                dframe_dict[cm] = dframe.append(h5pyframe)
                    
    finalmethods = [
        "no PP -- no MS -- Pearson MS", "no PP --  1 MS -- Pearson MS", "no PP --  2 MS -- Pearson MS", "   PP -- no MS -- Pearson MS", "   PP --  1 MS -- Pearson MS", "   PP --  2 MS -- Pearson MS",
        "no PP -- no MS -- Cosine MS", "no PP --  1 MS -- Cosine MS", "no PP --  2 MS -- Cosine MS", "   PP -- no MS -- Cosine MS", "   PP --  1 MS -- Cosine MS", "   PP --  2 MS -- Cosine MS",
        "no PP -- no MS -- Angular MS", "no PP --  1 MS -- Angular MS", "no PP --  2 MS -- Angular MS", "   PP -- no MS -- Angular MS", "   PP --  1 MS -- Angular MS", "   PP --  2 MS -- Angular MS",
        "no PP -- no MS -- SSIM MS", "no PP --  1 MS -- SSIM MS", "no PP --  2 MS -- SSIM MS", "   PP -- no MS -- SSIM MS", "   PP --  1 MS -- SSIM MS", "   PP --  2 MS -- SSIM MS",
        "no PP -- no MS -- MFS Max MS", "no PP --  1 MS -- MFS Max MS", "no PP --  2 MS -- MFS Max MS", "   PP -- no MS -- MFS Max MS", "   PP --  1 MS -- MFS Max MS", "   PP --  2 MS -- MFS Max MS",
        "no PP -- no MS -- Shared Pixel", "   PP -- no MS -- Shared Pixel",
        "no PP -- no MS -- Hypergeometric", "   PP -- no MS -- Hypergeometric",
        "no PP -- no MS -- Contingency", "   PP -- no MS -- Contingency",
        "no PP -- no MS -- Local Std", "   PP -- no MS -- Local Std",
        "no PP -- no MS -- IntMagAn", "   PP -- no MS -- IntMagAn",
        "no PP -- no MS -- Grad Info", "   PP -- no MS -- Grad Info",
        "no PP -- no MS -- Mutual Info", "   PP -- no MS -- Mutual Info",
        "no PP -- no MS -- Histogram", "   PP -- no MS -- Histogram"
    ]

    index_prettify = {
        "no PP -- no MS -- Pearson MS": "     0 MS - Pearson", 
        "no PP --  1 MS -- Pearson MS": "     1 MS - Pearson", 
        "no PP --  2 MS -- Pearson MS": "     2 MS - Pearson", 
        "   PP -- no MS -- Pearson MS": "PP - 0 MS - Pearson", 
        "   PP --  1 MS -- Pearson MS": "PP - 1 MS - Pearson", 
        "   PP --  2 MS -- Pearson MS": "PP - 2 MS - Pearson",
        "no PP -- no MS -- Cosine MS": "     0 MS - Cosine", 
        "no PP --  1 MS -- Cosine MS": "     1 MS - Cosine", 
        "no PP --  2 MS -- Cosine MS": "     2 MS - Cosine", 
        "   PP -- no MS -- Cosine MS": "PP - 0 MS - Cosine", 
        "   PP --  1 MS -- Cosine MS": "PP - 1 MS - Cosine", 
        "   PP --  2 MS -- Cosine MS": "PP - 2 MS - Cosine",
        "no PP -- no MS -- Angular MS": "     0 MS - Angular",
        "no PP --  1 MS -- Angular MS": "     1 MS - Angular", 
        "no PP --  2 MS -- Angular MS": "     2 MS - Angular", 
        "   PP -- no MS -- Angular MS": "PP - 0 MS - Angular", 
        "   PP --  1 MS -- Angular MS": "PP - 1 MS - Angular", 
        "   PP --  2 MS -- Angular MS": "PP - 2 MS - Angular",
        "no PP -- no MS -- SSIM MS": "     0 MS - SSIM", 
        "no PP --  1 MS -- SSIM MS": "     1 MS - SSIM", 
        "no PP --  2 MS -- SSIM MS": "     2 MS - SSIM", 
        "   PP -- no MS -- SSIM MS": "PP - 0 MS - SSIM", 
        "   PP --  1 MS -- SSIM MS": "PP - 1 MS - SSIM", 
        "   PP --  2 MS -- SSIM MS": "PP - 2 MS - SSIM",
        "no PP -- no MS -- MFS Max MS": "     0 MS - MFS Max",
        "no PP --  1 MS -- MFS Max MS": "     1 MS - MFS Max", 
        "no PP --  2 MS -- MFS Max MS": "     2 MS - MFS Max", 
        "   PP -- no MS -- MFS Max MS": "PP - 0 MS - MFS Max", 
        "   PP --  1 MS -- MFS Max MS": "PP - 1 MS - MFS Max", 
        "   PP --  2 MS -- MFS Max MS": "PP - 2 MS - MFS Max",
        "no PP -- no MS -- Shared Pixel": "     0 MS - Shared Pixel", 
        "   PP -- no MS -- Shared Pixel": "PP - 0 MS - Shared Pixel",
        "no PP -- no MS -- Hypergeometric": "     0 MS - Hypergeometric", 
        "   PP -- no MS -- Hypergeometric": "PP - 0 MS - Hypergeometric",
        "no PP -- no MS -- Contingency": "     0 MS - Contingency", 
        "   PP -- no MS -- Contingency": "PP - 0 MS - Contingency",
        "no PP -- no MS -- Local Std": "     0 MS - Local Std", 
        "   PP -- no MS -- Local Std": "PP - 0 MS - Local Std",
        "no PP -- no MS -- IntMagAn": "     0 MS - IntMagAn", 
        "   PP -- no MS -- IntMagAn": "PP - 0 MS - IntMagAn",
        "no PP -- no MS -- Grad Info": "     0 MS - Grad Info", 
        "   PP -- no MS -- Grad Info": "PP - 0 MS - Grad Info",
        "no PP -- no MS -- Mutual Info": "     0 MS - Mutual Info", 
        "   PP -- no MS -- Mutual Info": "PP - 0 MS - Mutual Info",
        "no PP -- no MS -- Histogram": "     0 MS - Histogram", 
        "   PP -- no MS -- Histogram": "PP - 0 MS - Histogram"
    }
    
    stats_prettify = {
        "silhouette_score(h)": "SCS Val",
        "silhouette_score(h) ranks": "SCS Rank",
        "calinski_harabasz_score(h)": "CHI Val",
        "calinski_harabasz_score(h) ranks": "CHI Rank",
        "max_clustersize_ratio": "MCR",
        "global_min_max_ratio_99_1(l)": "R-99-1",
        "global_min_max_ratio_95_5(l)": "R-95-5",
    }

    for cluster_method, dframe in dframe_dict.items():
        dframe_dict[cluster_method] = dframe.loc[finalmethods]
        
        ranked_ss = dframe_dict[cluster_method]["silhouette_score(h)"].rank(ascending=True)  / dframe_dict[cluster_method].shape[0]
        dframe_dict[cluster_method].insert(loc=1, column="silhouette_score(h) ranks", value=ranked_ss)

        ranked_chi = dframe_dict[cluster_method]["calinski_harabasz_score(h)"].rank(ascending=True) / dframe_dict[cluster_method].shape[0]
        dframe_dict[cluster_method].insert(loc=3, column="calinski_harabasz_score(h) ranks", value=ranked_chi)

        rank_sum = dframe_dict[cluster_method][["silhouette_score(h) ranks", "calinski_harabasz_score(h) ranks"]].sum(axis=1) / 2
        dframe_dict[cluster_method].insert(loc=7, column="Score", value=rank_sum)

        dframe_dict[cluster_method].rename(index=index_prettify, inplace=True)
        dframe_dict[cluster_method].rename(columns=stats_prettify, inplace=True)

    print("Start HTML creation ..... ", end = "")
    style = [dict(selector="", props=[("border-collapse","collapse")])]
    
    soup = BeautifulSoup("<html><head><style></style></head><body><h3></h3><div class=container></div><h3></h3><div class=container></div><h3></h3><div class=container></div><h3></h3><div class=container></div></body></html>", 'html.parser')
    soup.style.append(".container{display: flex;}")
    soup.style.append("table{margin: 0.2vw; margin-bottom:2vw; margin-top:-0.6vw; width: 34vw; min-width: 34vw;}")
    soup.style.append("caption{font-weight: bold;}")
    soup.style.append("hr{margin-left:0.2vw; margin-top:3vw; margin-bottom:5vw; width: 90.6vw;}")
    soup.style.append("table td{font-size: 13pt !important;}")

    #html_order = ["hierarchical", "affinity-propagation", "kmedoids-em", "kmedoids-pam", "spectral-clustering", "dbscan", "optics"]
    title_prettify = {
        "HierarchicalClustering": "Hierarchical Clustering",
        "AffinityPropagationClustering": "Affinity-Propagation",
        "MsiCommunityDetectionPCA": "MSI-Community-Detection-PCA",
        "MsiCommunityDetectionStatistics": "MSI-Community-Detection-Statistics"
    }

    limit = limit
    #print(dframe_dict.items())
    for idx, (cluster_method, dframe) in enumerate(dframe_dict.items()):
        #print(idx)
        soup.find_all('h3')[idx].append(title_prettify[cluster_method])
        sub_frame = dframe[["SCS Rank", "CHI Rank", "Score", "SCS Val", "CHI Val"]]
        sub_frame = sub_frame.sort_values(by="Score", ascending=False)
        sub_frame = sub_frame.iloc[:limit]
        sub_frame.insert(loc=0, column="Method", value=sub_frame.index)
        html = (sub_frame.style.hide_index().set_table_styles(style).set_properties(**{'font-size': '12pt', 'font-family': 'Calibri', 'border': '2px solid black', 'border-collapse': 'collapse', 'border-spacing':'0px'}).bar(axis=0, color='orange', vmin=0).render())
        broth = BeautifulSoup(html, 'html.parser')
        soup.find_all('div')[idx].append(broth.table)
        soup.style.append(broth.style.string)

    soup.style.append(".col0{min-width:12vw !important;, width:12vw !important;}")
    soup.style.append(".col1{min-width:4vw !important; width:4vw !important;}")
    soup.style.append(".col2{min-width:4vw !important; width:4vw !important;}")
    soup.style.append(".col3{min-width:6vw !important; width:6vw !important;}")
    soup.style.append(".col4{min-width:4vw !important; width:4vw !important;}")
    soup.style.append(".col5{min-width:4vw !important; width:4vw !important;}")
    
    for tag in soup.find_all("td", class_="col0"):
        content = tag.string
        tag.clear()
        pre_tag = soup.new_tag("pre")
        pre_tag.string = content
        tag.append(pre_tag)

    soup.style.append("pre{margin:0 !important; font-size: 12pt !important; font-family: Lucida Console, Courier, monospace;}")
    
    with open(savepath, "w") as f:
        f.write(str(soup))
        f.close()

    print("finished!")