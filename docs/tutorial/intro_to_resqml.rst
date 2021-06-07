Introduction to Reservoir Modelling with RESQML
===============================================

What is RESQML?
---------------
RESQML is a standard format for storing and exchanging reservoir models. It is 'software vendor neutral' meaning that the standard is not defined or owned by a single commercial company. Instead, RESQML is a public domain standard defined by Energistics, which is a consortium of oil companies and reservoir software companies.

The RESQML standard aims to be:

* Comprehensive: covering all the main elements of subsurface models for the entire modelling workflow
* Flexible: parts of a model can themselves constitute a valid package of data
* Efficient: array data is stored in binary form
* Rigorous: units and coordinate reference systems are thorough, and unique identifiers ensure correct identification of parts

RESQML is *the* standard for subsurface modelling, gradually being adopted across the industry.

Energistics also defines two other, related, standards: PRODML covers production data and WITSML handles well data.

Current Version
^^^^^^^^^^^^^^^
The current recommended version (as of March 2021) of the RESQML standard is 2.0.1. Version 2 marked a major developmentfrom earlier versions.

The Structure of a RESQML Model
-------------------------------
Physically, a resqml model is stored in a compressed file with the extension .epc together (usually) with one or more hdf5 format files holding array data, with the extension .h5. Neither of these file types can be viewed or edited using a simple text editor – other more specialised tools must be used.

The *epc* File
^^^^^^^^^^^^^^
The *epc* (Energistics Package Convention) file is the 'main' file holding the metadata for the model, along with scalar data. It also contains links to any hdf5 (.h5) files holding array data for the model. The *epc* file itself conforms to another standard file structure (not specific to Energistics) known as opc. It is basically a zipped (compressed) file containing a set of xml files. An individual xml file might be one of the following (amongst others):

* The main contents file holding a list of the other xml files.
* A primary model part, holding data for an object such as a coordinate reference system (crs), a surface, a grid, a grid property etc.
* A relationship file, holding the relationships between a model part and other parts, for example which crs applies to the geometry of a grid object. (These relationship files are grouped together in a subdirectory named ``_rels`` within the zipped *epc* file structure.)
* A reference to an *hdf5* file holding array data for one or more of the model parts.
* A documentation folder holding non-data documents that are not defined by the standard.

The RESQML standard does not specify a minimum set of objects, or parts, for a model to be valid. So, for example, a coordinate reference system on its own would be a valid (if minimal) RESQML model (or package, to use the Energistics term). For maximum flexibility, there is not even a requirement that all the objects referred to in the relationships must be present. So, for example, if a grid object refers to a crs, then that crs may be absent. This allows a partial model to be transferred, for example, when only a small part of a model has changed.

Because array data is not stored in the epc file, and because it uses a compressed format, file sizes are small – perhaps a few tens of kilobytes for a sizeable model.

The bulk of the RESQML standard is published in the form of xml schema definition files (with extension xsd). These files specify the required xml contents for each of the types of object.

hdf5 (.h5) Files
^^^^^^^^^^^^^^^^
Any array data that forms part of the model must be stored in an hdf5 file, and not within one of the xml files within the epc (though those xml files will contain references to the hdf5 files). The hdf5 format – hierarchical data format – is not specific to Energistics. It is widely used in high performance technical computing. It stores array data in binary format and can handle extremely large data sets. A single hdf5 file can hold multiple arrays, organised within the file in a hierarchical structure rather like a directory structure. Random access to an array, or part of an array, within an hdf5 file is fast. The detailed format of the array storage is highly compatible with python numpy arrays.

All the arrays for a RESQML model may be stored in a single hdf5 file, or they may be spread amongst multiple files. Furthermore, an hdf5 file may be referred to by more than one *epc* file, potentially reducing the duplication of data. A particular *epc* file does not need to refer to all the arrays within an hdf5 file. Despite all this flexibility, the recommendation is to keep a simple one-to-one correspondence between *epc* and hdf5 files wherever possible (and this is the default behaviour of the Bifröst code base).

The hdf5 files can be large: typically several gigabytes.

RESQML Objects and Universally Unique Identifiers
-------------------------------------------------
From the discussion above, it can be seen that a RESQML model is a collection of parts, most of which are RESQML objects. The relationships between these objects also forms part of the model. The RESQML standard defines many classes of object, such as a fault interpretation or an IJK grid. There is no limit on how many objects of any given type are included in a single RESQML model. For example, several different coodinate reference systems could be included in one model.

To help keep track of these objects in a rigorous way, whenever a new object is created, or modified, it is given a new universally unique identifier (UUID), also known as a globally unique identifier (GUID). The format of the UUIDs is not specific to Energistics – it is the subject of an ISO standard (ISO/IEC 9824-8). As the name suggests, a UUID is completely unique. If you see 2 UUIDs that are the same, they are referring to the same thing. It is this feature that allows partial RESQML models to be moved around and joined up again correctly.

A UUID is actually a 128 bit integer. However, it is usually displayed in hexadecimal form with hyphens at key points, for example: decd627f-c91e-47e1-946e-8a6a4d91617f

All the links between objects within a RESQML model use the UUIDs as the way to ensure that the correct objects are identified. They also make it possible to create a rigorous audit trail of the development of a model, though such an audit trail is not currently covered by the RESQML standard.

Units of Measure
----------------
The Energistics standards include a rigorous handling of units of measure (uom). This aspect of the standards is common to RESQML, PRODML and WITSML. The uom definitions include things such as:

* A (very long) list of physical units in use around the world.
* Which units are convertible to each other, together with conversion factors.
* A (long) list of quantity classes, such as rock permeability.
* Which units are applicable to which quantity classes.
* The fundamental dimensions of a quantity class (in terms of Mass, Length, Time etc.)

Although the Energistics UoM definitions are primarily intended to support the Energistics standards, they are general purpose and could form the basis of any technical unit handling and conversion system.

Software Vendors and RESQML
---------------------------
Eventually, all the companies producing software for reservoir modelling will be reading and writing RESQML models as standard. However, it is a large and rather complex standard so, in practice, most companies will start with a partial implementation and build up to a full implementation over a course of years. Hopefully we will relatively quickly reach a stage where there is sufficient common functionality for RESQML models to be a practical way of moving models between different packages. Longer term, it is hoped that a RESQML model will be seen as the reservoir model, with temporary forays into other formats as needed during the workflow.

The current state of play of the main providers of modelling software to BP is:

* Emerson (RMS): From version 12, RMS has some RESQML import and export capability. Not all RESQML classes are yet supported. However, plug-ins are being released on a regular basis with new RESQML functionality added each time. There is a close working relationship between bp and Emerson.
* Schlumberger (Petrel): There is an add on module available which allows Petrel to import and export in RESQML format. A successful export has been made within BP, though not all object types are included in the RESQML model.
* Halliburton / Landmark (Nexus): Nexus cannot currently read or write RESQML models. However, development is underway to allow grids, grid properties and some other inputs to be sourced directly from a RESQML dataset, using a syntax similar to that for reading from a vdb. There is a good working relationship between bp and the Halliburton developers.
* DGI (Coviz): The Coviz API includes a RESQML import function that will convert from RESQML to DGI's proprietary c3grd format. It also includes an export function for converting in the opposite direction. It works well for IJK Grid geometries, though there appear to be some bugs needing to be fixed. It can handle coordinate reference systems. Although it can also process grid properties, it does so in a rudimentary manner that will need to be enhanced. (It currently does not use the standard RESQML property kinds or units of measure.) BP has a close working relationship with DGI and they are the most responsive of our software providers.
* F2I (FESAPI): You may not have heard of this small software consultancy. However, the latter 3 companies listed above are using the FESAPI interface to read and write RESQML models. This is the only API (application programming interface) available as an alternative to in-house development. It consists mainly of C++ code generated from the RESQML xsd schema files, providing conversion between C++ objects and RESQML xml trees. It primarily targets a Windows runtime environment. SWIG wrappers are available for onward conversion to Java (and could potentially be used to support Python). However, standard Python libraries provide much of the functionality to convert between xml and Python objects directly, as well as good hdf5 to numpy interaction, so FESAPI is not an ideal solution if working in Python.