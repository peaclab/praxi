COLUMBUS : Automated Software Discovery
=========================================

Usage:
-------------------

```bash
Usage: python columbus.py -f <config_file> {--list | --discover } --route {bin_names | file_paths | func_names|    docker_history | pack_manager}} <image_path>

Options:
  -h, --help            show this help message and exit
  -l, --list            list all the layers
  -d, --discover        Discover software from     the image.
  -r ROUTE, --route=ROUTE
                        specify discovery technique.     Choose from {bin-
                        names, file-paths, func-names, docker-history, pack-
                        manager}
  -f CFGFILE            Config file
```

Example:
-------------------
```bash
# python columbus.py -f config.cfg --discover --route bin_names /home/nadgowdas/dockerimages/images/mongodb.tar.gz 
LAYERID	                                                           LAYER-INDEX	SOFTWARE TAGS
-------	                                                               -------	------------
768d4f50f65f00831244703e57f64134771289e3de919a576441c9140e037ea2	1	[]
0be66d84b202b9cb7bd9a076c65860b7b031faab6be3b34b4b0087367a7e821f	2	['li', 'in', 'mk']
6656872b134b239f923f69dc996c0181c9ef9f4da9b6acbe66b039297ebbc232	3	[]
a096f8d6b7453d598f9dd3dcea5004e950733ee41144b0f7a92e17c47182f923	4	[]
06dc4c32d926a78cf401bf9e9343f91af95e393e6680fedc9c520a465102105d	5	[]
bf2b91535bb48c70cba486372704b8fba1b72799864577bc1d24eb293fe59c51	6	[]
5ee024791a26ec24ede1d459b3e7cb8091d28ef4d82850d89ab73916c8e16423	7	['pod', 'pod2', 'py']
60720a8081e0c0679baf9330d85dfc16089ddff07d93449eac577335f68437b9	8	['mongo', 'ld', 'iconv']


# time python columbus.py -f config.cfg --discover --route file_paths /home/nadgowdas/dockerimages/images/golang.tar.gz 
LAYERID	                                                           LAYER-INDEX	SOFTWARE TAGS
-------	                                                               -------	------------
768d4f50f65f00831244703e57f64134771289e3de919a576441c9140e037ea2	1	[]
0be66d84b202b9cb7bd9a076c65860b7b031faab6be3b34b4b0087367a7e821f	2	['man', 'zone', 'posix']
6656872b134b239f923f69dc996c0181c9ef9f4da9b6acbe66b039297ebbc232	3	[]
a096f8d6b7453d598f9dd3dcea5004e950733ee41144b0f7a92e17c47182f923	4	[]
06dc4c32d926a78cf401bf9e9343f91af95e393e6680fedc9c520a465102105d	5	[]
bf2b91535bb48c70cba486372704b8fba1b72799864577bc1d24eb293fe59c51	6	[]
5ee024791a26ec24ede1d459b3e7cb8091d28ef4d82850d89ab73916c8e16423	7	['py', 'per', 'python2.7']
9630bdb90f92d2d21a28e3225a7b7839a33488d6902c4b9075f3c4a1f42cb9ac	8	['go', 'do', 'golang-']

```
