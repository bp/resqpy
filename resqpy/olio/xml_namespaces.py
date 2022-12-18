"""xml_namespaces.py: Module defining constant resqml xml namespaces."""

# major revamp to support bespoke serialisation code, needed to emulate gSOAP, hence FESAPI

namespace = {}

namespace['xsd'] = 'http://www.w3.org/2001/XMLSchema'
namespace['xsi'] = 'http://www.w3.org/2001/XMLSchema-instance'
# namespace['schema'] = 'http://www.w3.org/2001/XMLSchema-instance'

namespace['eml'] = 'http://www.energistics.org/energyml/data/commonv2'
namespace['eml20'] = 'http://www.energistics.org/energyml/data/commonv2'
namespace['eml23'] = 'http://www.energistics.org/energyml/data/commonv2'
namespace['resqml2'] = 'http://www.energistics.org/energyml/data/resqmlv2'
namespace['resqml'] = 'http://www.energistics.org/energyml/data/resqmlv2'
namespace['resqml20'] = 'http://www.energistics.org/energyml/data/resqmlv2'
namespace['rels_ext'] = 'http://schemas.energistics.org/package/2012/relationships/'

namespace['rels'] = 'http://schemas.openxmlformats.org/package/2006/relationships'
namespace['rels_md'] = 'http://schemas.openxmlformats.org/package/2006/relationships/metadata/'

namespace['content_types'] = 'http://schemas.openxmlformats.org/package/2006/content-types'
namespace['cp'] = 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'

namespace['dcterms'] = 'http://purl.org/dc/terms/'
namespace['dc'] = 'http://purl.org/dc/elements/1.1/'

curly_namespace = {}
for key, url in namespace.items():
    curly_namespace[key] = '{' + url + '}'

inverse_namespace = {}
for key, url in namespace.items():
    if url not in inverse_namespace:
        inverse_namespace[url] = key


def colon_namespace(url):
    """Returns the short form namespace for the url, complete with colon suffix."""

    if url[0] == '{':
        return inverse_namespace[url[1:-1]] + ':'
    return inverse_namespace[url] + ':'
