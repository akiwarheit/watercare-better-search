import scrapy
from scrapy.spiders import SitemapSpider
import json
import os

class SitemapSpider(SitemapSpider):
    name = "sitemap_spider"
    sitemap_urls = ['https://www.watercare.co.nz/sitemap.xml']
    
    results = []

    def parse(self, response):
        details_texts = scrapy.Selector(response=response).css('main details *::text').getall()
        details_text = "\n".join(details_texts).strip() if details_texts else None

        if details_text:
            data = {
                'url': response.url,
                'content': details_text
            }
            self.results.append(data)
            
    def closed(self, reason):
        output_file = '../output/results.json'
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        
        self.log(f'Saved results to {output_file}')
