# -*- coding: utf-8 -*-
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import numpy as np
import pandas as pd

class WorldCloud(object):
    def __init__(self):
        pass

    def worldcloud(self):

        df = pd.read_csv('~/Desktop/NLP/data/last.csv', encoding='utf-8', na_values='NULL')
        df = df.dropna()
        word_list = []
        for ix, i in enumerate(df['content']):
            i.encode('utf-8')
            word_list.append(i)

        wordlist_space_split = ' '.join(word_list)
        d = os.path.dirname(__file__, )
        font=os.path.join(os.path.dirname(__file__), "DroidSansFallbackFull.ttf")

        alice_coloring = np.array(Image.open(os.path.join(d,'e0f057b7a1a61de962d89347b6d7201f-d4o1tzm.jpg')))

        my_wordcloud = WordCloud(background_color='#EDEDED',
                                 max_words=100,
                                 font_step=1,
                                 mask=alice_coloring,
                                 random_state= 100,
                                 max_font_size=150,
                                 font_path=font,
                                 )
        my_wordcloud.generate(wordlist_space_split)

        image_colors = ImageColorGenerator(alice_coloring)

        plt.show(my_wordcloud.recolor(color_func=image_colors))
        plt.imshow(my_wordcloud)
        plt.axis('off')
        plt.show()

        my_wordcloud.to_file(os.path.join(d, 'Pairs_accord_colors_cloud.jpg'))

if __name__ == '__main__':
    res = WorldCloud()
    res.worldcloud()