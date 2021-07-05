Working with a Single Property
==============================

The previous tutorial looked at working with sets of property arrays, using the resqpy PropertyCollection class. This tutorial explores an alternative when working with a single property array. The resqpy Property class behaves more like the other resqpy high level object classes. In particular, one resqpy Property object corresponds to one RESQML object of class ContinuousProperty, DiscreteProperty or CategoricalProperty.

Accessing a property array for a given uuid
-------------------------------------------
Assuming that an existing RESQML dataset has been opened, we can find the uuid of a particular property using the familiar methods from the Model class. For example:

.. code-block:: python

    blocked_well_uuid = Model.uuid(obj_type = 'BlockedWellboreRepresentation', title = 'INJECTOR_3')
    assert blocked_well_uuid is not None
    kh_uuid = Model.uuid(obj_type = 'ContinuousProperty', title = 'KH', related_uuid = blocked_well_uuid)

It is a good idea to include the *related_uuid argument*, as shown above, to ensure that the property is 'for' the required representation object (in this example a blocked well). Having determined the uuid for a property, we can instantiate the corresponding resqpy object in the familiar way:

.. code-block:: python

    import resqpy.property as rqp
    inj_3_kh = rqp.Property(model, uuid = kh_uuid)

To get at the actual property data as a numpy array, use the *array_ref()* method:

.. code-block:: python

    inj_3_kh_array = inj_3_kh.array_ref()

The *array_ref()* method has some optional arguments for coercing the array element data type (dtype), and for optionally returning a masked array.

The Property class includes similar methods to PropertyCollection for retrieving the metadata for the property. As the Property object is for a single property, the '_for_part' is dropped from the method names, as is the part argument. For example:

.. code-block:: python

    assert inj_3_kh.property_kind() == 'permeability thickness'
    assert inj_3_kh.indexable_element() == 'cells'
    kh_uom = inj_3_kh.uom()

Behind the scenes, the Property object has a singleton PropertyCollection class as an attribute. This can be used by calling code if access to other PropertyCollection functionality is needed. Here is an example that sets up a normalised version of the array data:

.. code-block:: python

    normalized_inj_3_kh = inj_3_kh.collection.normalized_part_array(inj_3_kh.part, use_logarithm = True)

When to use PropertyCollection and when Property
------------------------------------------------
The main advantage of the PropertyCollection class is that it allows identification of individual property arrays or groups of property arrays based on the metadata items. In particular, the property kind and facet information is a preferable way to track down a particular property than relying on the citation title.

On the other hand, once the uuid for a particular property has been established, it can be passed around in code and used to instantiate an individual Property object when required. This is simpler than having to deal with a PropertyCollection once the individual uuid is known.
