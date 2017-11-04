# coding:utf-8
import requests
import json
import re
import pandas as pd
hearder = {
'accept':'application/json, text/plain, */*',
'authorization':'Bearer 2|1:0|10:1504926435|4:z_c0|92:Mi4xWUdzSUFnQUFBQUFBQUFKd0hlVE9DeVlBQUFCZ0FsVk40LWZhV1FDRzdfdlFlM0xJSW42cVpaM3QxWkkyb0V6Z01R|a8a5631d9225ad2b59183fdfc506be642836e08de1821bce430d826835258837',
'Connection':'keep-alive',
'Host':'www.zhihu.com',
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
}

class ZhihuSpider(object):

    def __init__(self, pages, num):
        self.pages = pages
        self.num = num

    def spider_main(self):
        res_list = []
        for n in range(0, self.pages+20, 20):
            print(u'开始第', int(n/20)+1,u'页')
            url = "https://www.zhihu.com/api/v4/questions/{}/answers?include=data%5B*%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cupvoted_followees%3Bdata%5B*%5D.mark_infos%5B*%5D.url%3Bdata%5B*%5D.author.follower_count%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics&offset={}&limit=20&sort_by=default".format(self.num, n)
            json_ = self.get_json(url)
            length = len(self.get_json(url)['data'])
            for i in range(length):
                json_text = json_['data'][i]
                question_info = json_text.get('content')
                author = json_text.get('author').get('name')
                content = re.sub(r'(<.{1,2}>)', '', question_info)
                print(author,content)

                res_list.append([author,'|',  content])
        df = pd.DataFrame(res_list, columns=['name', 'content'])
        df.to_csv('冈波仁齐.csv', encoding='utf-8')


    @staticmethod
    def get_json(url):
        # print(json.loads(requests.get(url, headers=hearder).text))
        return json.loads(requests.get(url, headers=hearder).text)


if __name__ == '__main__':
    # 填入回答网址编号num，总回答数pages
    res = ZhihuSpider(pages=1165, num=61112615)
    res.spider_main()



