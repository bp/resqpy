

def iter_wellbore_frames(self):
    """Iterable of all WellboreFrames associated with a trajectory.

    Yields:
       frame: instance of :class:`resqpy.organize.WellboreFrame`

    :meta common:
    """
    uuids = self.model.uuids(obj_type = "WellboreFrameRepresentation", related_uuid = self.uuid)
    for uuid in uuids:
        yield WellboreFrame(self.model, uuid = uuid)