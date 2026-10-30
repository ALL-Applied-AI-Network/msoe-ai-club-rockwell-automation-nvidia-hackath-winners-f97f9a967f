"""Fill-order decisions; visual occupancy alone cannot prove completion or routing."""
CLASSES={'pass','missing_cap','crooked_cap','damaged_vial','no_vial','uncertain'}


def summarize(slots, counts=None, complete=False):
    if len(slots)!=60 or {s['id'] for s in slots}!={f'{t}{i:02d}' for t in 'GS' for i in range(1,31)}:
        raise ValueError('Expected all 60 unique fixture positions')
    if any(s['class'] not in CLASSES or s['position']!=int(s['id'][1:]) for s in slots):
        raise ValueError('Invalid slot classification or position')
    counts=counts or {};groups={}
    for prefix,area in [('G','good'),('S','scrap')]:
        items=sorted((s for s in slots if s['id'][0]==prefix),key=lambda x:x['position'])
        expected=counts.get(area)
        if expected is not None and (type(expected)!=int or not 0<=expected<=30):
            raise ValueError('Placement count must be an integer between 0 and 30')
        # Completion establishes the good order's 30 placements; scrap need not be full.
        if complete and area=='good':expected=30
        occupied=[s['position'] for s in items if s['class'] not in ['no_vial','uncertain']]
        boundary=max(occupied+[expected or 0]);missing=[];gaps=[];pending=[];unknown=[]
        for s in items:
            i=s['position'];cls=s['class']
            if cls=='uncertain':s['action']='review';unknown.append(i)
            elif cls=='no_vial':
                if expected is not None and i<=expected:s['action']='missing';missing.append(i)
                elif i<=boundary:s['action']='possible_gap';gaps.append(i)
                else:s['action']='pending';pending.append(i)
            else:s['action']='accept' if cls=='pass' else 'defect'
        groups[area]=dict(occupied=len(occupied),capacity=30,expected_through=expected,
                          missing=missing,possible_gaps=gaps,pending=pending,uncertain=unknown,
                          visible_defects=[s['position'] for s in items if s['class'] in ['missing_cap','crooked_cap','damaged_vial']],
                          fill_boundary=boundary,order_complete_signal=complete and area=='good',
                          routing_verdict='unknown: upstream verdicts required' if area=='scrap' else None)
    return groups


class TemporalTracker:
    """Two consecutive observations suppress transient events; never infer a robot count."""
    def __init__(self):self.previous={};self.candidates={};self.streak={};self.frame=0
    def update(self,slots):
        self.frame+=1;events=[]
        for s in slots:
            key=s['id'];cls=s['class']
            self.streak[key]=self.streak.get(key,0)+1 if self.candidates.get(key)==cls else 1
            self.candidates[key]=cls
            if self.streak[key]>=2 and cls!='uncertain':
                old=self.previous.get(key)
                if old is not None and old!=cls:
                    kind='placement_observed' if old=='no_vial' and cls!='no_vial' else 'appearance_changed'
                    events.append(dict(frame=self.frame,position=key,event=kind,previous=old,current=cls))
                self.previous[key]=cls
            s['stable']=self.previous.get(key)==cls and self.streak[key]>=2
        return events
