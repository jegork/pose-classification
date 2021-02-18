import requests, tarfile, os

def fetch_videos(videos_dir='videos'):
    """Download, unpack and rename the videos (if not already), return the videos folder name."""
    if not os.path.isdir('./' + videos_dir):
        videos_link = 'https://www.robots.ox.ac.uk/~vgg/data/tv_human_interactions/data/tv_human_interactions_videos.tar.gz'
        r = requests.get(videos_link)
        open('videos.tar.gz', 'wb').write(r.content)

        tar = tarfile.open('videos.tar.gz', 'r:gz')
        tar.extractall()
        tar.close()

        os.rename('tv_human_interactions_videos', videos_dir)
        os.remove('videos.tar.gz')
        
    else:
        print('Files already downloaded!')
        
    return videos_dir
        
def separate_videos(videos_dir='videos'):
    """Separate the videos in videos_dir by classes and place them in separate folders."""
    base_path = videos_dir + '/'
    videos_list = os.listdir(base_path)
    
    videos = {}

    for name in videos_list:
        x = name.split("_")
        v_class = x[0].lower()
        v_id = x[1]

        if v_class not in videos:
            videos[v_class] = []
        videos[v_class] += [name]

    for v_class in videos.keys():
        if not os.path.exists(base_path+"/"+v_class):
            os.mkdir(base_path+"/"+v_class)

        for v in videos[v_class]:
            v_id = v.split("_")[1]

            os.rename(base_path+v, base_path+v_class+"/"+v_id)
