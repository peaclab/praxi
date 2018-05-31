from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q
import logging

ES_TYPE = 'fsmeta'

class ESClient():
    def __init__(self, server, port):
                #FIXME add try/catch
                self.client = Elasticsearch(
                    ["%s:%s"%(server,port)],
                    timeout=30, max_retries=10, retry_on_timeout=True
                )
                self.tracer = logging.getLogger('elasticsearch.trace')
                self.tracer.setLevel(logging.INFO)
                self.tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    def __insert_fs_metdata__(self, imageid, layerid, filelist):
                count = 0
                if not self.client.indices.exists(imageid):
                        self.client.indices.create(imageid)

                helpers.bulk(self.client, tuple(filelist))
                self.client.indices.refresh(index=imageid)

    def __get_all_binaries__(self,imageid, layerid):
                binlist = []
                #Search().using(client).query("match", type="directory").query("match", permission="*x*")
                query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", layer=layerid)\
                        .query("match", permission="x")
                #res = query.execute()
                res = query.scan()

                for hit in res:
                        binlist.append(str(hit.path))
                return binlist

    def get_all_matched_execs(self, imageid, layerid, tag):
                pathlist = []
                query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", path=tag)\
                        .query("match", layer=layerid)\
                        .query("match", permission="x")
                #res = query.execute()
                res = query.scan()
                for hit in res:
                        pathlist.append(str(hit.path))

                return pathlist

    def get_all_matched_files(self, imageid, layerid, tag):
                pathlist = []
                query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", path=tag)\
                        .query("match", layer=layerid)
                #res = query.execute()
                res = query.scan()
                for hit in res:
                        pathlist.append(str(hit.path))

                return pathlist

    def get_stats_by_layer(self, imageid, layerid):
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", layer=layerid)
        res = query.scan()
        #res = query.execute()
        size = 0
        count = 0
        for hit in res:
                count+=1
                if hit.type == 'file':
                        size+=hit.size

        return (size, count)

    def __get_all_files__(self, imageid, layerid):
                pathlist = []
                query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", layer=layerid)
                #res = query.execute()
                res = query.scan()
                for hit in res:
                    try:
                        pathlist.append(str(hit.path))
                    except UnicodeEncodeError:
                        #print hit.path
                        pass

                return pathlist

    def __get_files_count(self, imageid, layerid):
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", layer=layerid)
        res = query.scan()
        return len(res)

    def __get_dirs_count(self, imageid, layerid):
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="directory")\
                        .query("match", layer=layerid)
        res = query.scan()
        return len(res)

    def get_executables_stats(self, imageid, layerid, tag):
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", layer=layerid)\
                        .query("match", permission="x")
        res = query.scan()
        total_execs = sum(1 for _ in res)
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", path=tag)\
                        .query("match", layer=layerid)\
                        .query("match", permission="x")
        res = query.scan()
        matched_execs = sum(1 for _ in res)
        return (total_execs, matched_execs)

    def get_filepaths_stats(self, imageid, layerid, tag):
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", layer=layerid)
        res = query.scan()
        total_paths = sum(1 for _ in res)
        query = Search().using(self.client)\
                        .index(imageid) \
                        .query("match", type="file")\
                        .query("match", path=tag)\
                        .query("match", layer=layerid)
        res = query.scan()
        matched_paths = sum(1 for _ in res)
        return (total_paths, matched_paths)

    def __del_index__(self, index):
        self.client.indices.delete(index=index, ignore=[400, 404])


