import os, sys
import logging
import pynmrstar
import pint
import numpy as np

aa3to1 = {'CYS': 'C', 'GLN': 'Q', 'ILE': 'I', 'SER': 'S', 'VAL': 'V', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'LYS': 'K', 'THR': 'T', 'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ASP': 'D', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'GLU': 'E', 'TYR': 'Y'}
aa1to3 = {v:k for k,v in aa3to1.items()}

def get_tag_vals(sf, tag, warn=None, default=None, strip_str=False, indices=None, empty_val_str='.'):
    try:
        vals = sf.get_tag(tag)
        if strip_str:
            vals = [v.strip() for v in vals]
        if empty_val_str:
            vals = [v if v != empty_val_str else '' for v in vals]
        if indices is not None:
            if type(indices) == list:
                vals = [vals[i] for i in indices]
            else:
                vals = vals[indices]
        return vals
    except:
        if warn:
            logging.warning(warn)
        logging.debug(f'failed to retrieve tag {tag}')
    return default

class Entity(object):
    def __init__(self, sf):
        super(Entity, self).__init__()
        self.id = get_tag_vals(sf, '_Entity.ID', indices=0)
        self.name = get_tag_vals(sf, '_Entity.Name', indices=0)
        self.details = get_tag_vals(sf, '_Entity.Details', indices=0)
        self.type = get_tag_vals(sf, '_Entity.Type', indices=0)
        self.polymer_type = get_tag_vals(sf, '_Entity.Polymer_type', indices=0)
        self.polymer_type_details = get_tag_vals(sf, '_Entity.Polymer_type_details', indices=0)
        self.polymer_author_seq_details = get_tag_vals(sf, '_Entity.Polymer_author_seq_details', indices=0)
        self.seq = get_tag_vals(sf, '_Entity.Polymer_seq_one_letter_code', indices=0)
        if self.seq is not None and self.seq not in ['', '.']:
            self.seq = self.seq.replace('\n', '')
        else:
            self.seq = None
        self.fragment = get_tag_vals(sf, '_Entity.Fragment')
        self.weight = get_tag_vals(sf, '_Entity.Formula_weight')
        self.db_links = list(zip(
            get_tag_vals(sf, '_Entity_db_link.Database_code', default=[]),
            get_tag_vals(sf, '_Entity_db_link.Accession_code', default=[]),
            get_tag_vals(sf, '_Entity_db_link.Entry_mol_code', default=[]),
            get_tag_vals(sf, '_Entity_db_link.Seq_query_to_submitted_percent', default=[]),
            get_tag_vals(sf, '_Entity_db_link.Seq_identity', default=[])
            ))
    
    def __str__(self):
        if self.seq is not None:
            seq_str = f', seq="{self.seq[:10] + "..." if type(self.seq) == str and len(self.seq) > 13 else self.seq}"'
        else:
            seq_str = ""
        return f'(Entity {self.id}: name="{self.name}", type="{self.type}"{seq_str})'
    
    def __repr__(self):
        return f"<Entity {self.id}>"

class Assembly(object):
    def __init__(self, sf):
        super(Assembly, self).__init__()
        self.name = get_tag_vals(sf, '_Assembly.Name', indices=0)
        self.id = get_tag_vals(sf, '_Assembly.ID', indices=0)
        self.details = get_tag_vals(sf, '_Assembly.Details', indices=0)
        self.n_components = get_tag_vals(sf, '_Assembly.Number_of_components', indices=0)
        self.organic_ligands = get_tag_vals(sf, '_Assembly.Organic_ligands', indices=0)
        self.metal_ions = get_tag_vals(sf, '_Assembly.Metal_ions', indices=0)
        self.molecules_in_chemical_exchange = get_tag_vals(sf, '_Assembly.Molecules_in_chemical_exchange', indices=0)
        self.entities = list(zip(
            get_tag_vals(sf, '_Entity_assembly.ID', default=[]),
            get_tag_vals(sf, '_Entity_assembly.Entity_ID', default=[]),
            get_tag_vals(sf, '_Entity_assembly.Entity_label', default=[]),
            get_tag_vals(sf, '_Entity_assembly.Physical_state', default=[])
            ))
    
    def __str__(self):
        return f'(Assembly {self.id}: entities {[(e[0], e[1]) for e in self.entities]}, {self.n_components} components)'
    
    def __repr__(self):
        return f"<Assembly {self.id}>"

class SampleConditions(object):
    def __init__(self, sf):
        super(SampleConditions, self).__init__()
        self.id = get_tag_vals(sf, '_Sample_condition_list.ID', indices=0)
        self.ionic_strength = (None, None)
        self.pH = (None, None)
        self.pressure = (None, None)
        self.temperature = (None, None)
        info = list(zip(
            get_tag_vals(sf, '_Sample_condition_variable.Type', default=[]),
            get_tag_vals(sf, '_Sample_condition_variable.Val', default=[]),
            get_tag_vals(sf, '_Sample_condition_variable.Val_units', default=[])
            ))
        for t,val,unit in info:
            if 'ionic strength' in t.lower():
                self.ionic_strength = (val, unit)
            elif t.lower() == 'ph':
                self.pH = (val, unit)
            elif t.lower() == 'pressure':
                self.pressure = (val, unit)
            elif 'temp' in t.lower():
                self.temperature = (val, unit)
            else:
                logging.debug(f'Skipping sample condition {t} = {val} {unit}')
    
    def convert_unit(self, val, unit, target_unit):
        '''too slow...'''
        ureg = pint.UnitRegistry()
        try:
            factor = ureg.parse_expression(unit).to(target_unit).magnitude
        except:
            if target_unit == 'M' and val > 10:
                assumed_unit = 'mM'
            elif target_unit == 'K' and val < 150:
                assumed_unit = '°C'
            else:
                assumed_unit = target_unit
            logging.warning(f'Could not parse unit string for sample condition {self.id}: {val} "{unit}" , assuming {assumed_unit}')
            factor = ureg.parse_expression(assumed_unit).to(target_unit).magnitude
        return val * factor

    def convert_val(self, val):
        try:
            val = float(val)
        except:
            logging.warning(f'failed to convert value {val}')
            val = np.nan
        return val

    def get_pH(self):
        val = self.convert_val(self.pH[0])
        if np.isnan(val):
            logging.warning(f'No information on pH for sample condition {self.id}, assuming 7.0')
            return 7.0
        return val
    
    def get_temperature(self, fix_outliers=True):
        val = self.convert_val(self.temperature[0])
        if np.isnan(val):
            logging.warning(f'No information on temperature for sample condition {self.id}, assuming 298 K')
            return 298.
        if 'C' in self.temperature[1]:
            const0, const1, factor = 0., 273.15, 1.
        elif 'F' in self.temperature[1]:
            const0, const1, factor = -32., 273.15, 1.8
        elif self.temperature[1] == 'K':
            const0, const1, factor = 0., 0., 1.
        else:
            const0, const1, factor = 0., 0., 1.
            logging.info(f'Temperature unit unknown: {self.temperature[1]}, assuming K')
        if fix_outliers and const1 == 0.: # not explicitly °C or °F
            if 15 <= val < 50:
                logging.info(f'Very low temperature: {val}, assuming unit should be °C')
                const0, const1, factor = 0., 273.15, 1.
            elif 50 <= val <= 100:
                const0, const1, factor = -32., 273.15, 1.8
                logging.info(f'Low temperature: {val}, assuming unit should be °F')
        return (val + const0) * factor + const1
        #return self.convert_unit(val, self.temperature[1], 'K')
    
    def get_pressure(self):
        # TODO: implement
        return
        #return self.convert_unit(self.pressure[0], self.pressure[1], 'atm')
    
    def get_ionic_strength(self, fix_outliers=True):
        val = self.convert_val(self.ionic_strength[0])
        if np.isnan(val):
            logging.warning(f'No information on ionic strength for sample condition {self.id}, assuming 0.1 M')
            return 0.1
        if self.ionic_strength[1] == 'M':
            const1, factor = 0., 1.
        elif self.ionic_strength[1] == 'mM':
            const1, factor = 0., 0.001
        elif self.ionic_strength[1] == 'mu':
            const1, factor = 0., 0.000001
        else:
            const1, factor = 0., 1.
            logging.info(f'Ionic strength unit unknown: {self.ionic_strength[1]}, assuming K')
        if fix_outliers and factor == 1.:
            if val > 5:
                logging.info(f'High ionic strength: {val}, assuming unit should be mM')
                factor = 0.001
        return val * factor + const1
        #return self.convert_unit(val, self.ionic_strength[1], 'M')

    def __str__(self):
        return f'(Conditions {self.id}: pH {self.get_pH()}, {self.get_temperature()} K, {self.get_ionic_strength()} M)'
    
    def __repr__(self):
        return f"<Conditions {self.id}>"

class Sample(object):
    def __init__(self, sf):
        super(Sample, self).__init__()
        self.id = get_tag_vals(sf, '_Sample.ID', indices=0)
        self.type = get_tag_vals(sf, '_Sample.Type', indices=0)
        self.components = list(zip(
            get_tag_vals(sf, '_Sample_component.ID', default=[]),
            get_tag_vals(sf, '_Sample_component.Assembly_ID', default=[]),
            get_tag_vals(sf, '_Sample_component.Entity_ID', default=[]),
            get_tag_vals(sf, '_Sample_component.Mol_common_name', default=[]),
            get_tag_vals(sf, '_Sample_component.Concentration_val', default=[]),
            get_tag_vals(sf, '_Sample_component.Concentration_val_units', default=[])
            ))
    
    def __str__(self):
        return f'(Sample {self.id}: type {self.type}, components: {[(c[0],c[1],c[2]) for c in self.components]})'
    
    def __repr__(self):
        return f"<Sample {self.id}>"

class ShiftTable(object):
    def __init__(self, sf):
        super(ShiftTable, self).__init__()
        self.id = get_tag_vals(sf, '_Assigned_chem_shift_list.ID', indices=0)
        self.conditions = get_tag_vals(sf, '_Assigned_chem_shift_list.Sample_condition_list_ID', indices=0)
        self.experiments = list(zip(
            get_tag_vals(sf, '_Chem_shift_experiment.Experiment_ID', default=[]),
            get_tag_vals(sf, '_Chem_shift_experiment.Experiment_name', default=[]),
            get_tag_vals(sf, '_Chem_shift_experiment.Sample_ID', default=[]),
            get_tag_vals(sf, '_Chem_shift_experiment.Sample_state', default=[])
            ))
        self.shifts = list(zip(
            get_tag_vals(sf, '_Atom_chem_shift.Entity_assembly_ID', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Entity_ID', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Seq_ID', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Comp_ID', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Atom_ID', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Atom_type', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Val', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Val_err', default=[]),
            get_tag_vals(sf, '_Atom_chem_shift.Ambiguity_code', default=[])
            ))
        shifts = {}
        for s in self.shifts:
            if (s[0], s[1]) not in shifts:
                shifts[(s[0], s[1])] = []
            shifts[(s[0], s[1])].append(s)
        self.shifts = shifts
    
    def __str__(self):
        return f'(ShiftTable {self.id}: conditions {self.conditions}, shifts: {[(key, len(vals)) for key,vals in self.shifts.items()]})'
    
    def __repr__(self):
        return f"<ShiftTable {self.id}>"

class BmrbEntry(object):
    def __init__(self, id_, bmrb_dir):
        super(BmrbEntry, self).__init__()
        self.source = None
        
        self.id = id_
        self.type = None
        self.submission_date = None
        self.nmr_star_version = None
        self.original_nmr_star_version = None
        self.exp_method = None
        self.exp_method_subtype = None
        self.struct_keywords = []
        self.citation_title = None
        self.citation_journal = None
        self.citation_PubMed_ID = None
        self.citation_DOI = None
        self.citation_keywords = []
        self.assemblies = {}
        self.db_links = []
        self.n_components = None
        self.n_entities = None
        self.entities = {} # Physical_state, Name, Type, [(Database_code, Accession_code), ...]
        self.conditions = {}
        self.shift_tables = {}
        
        self.entry_path = os.path.join(bmrb_dir, f"bmr{id_}")
        fn3 = os.path.join(self.entry_path, f"bmr{id_}_3.str")
        if not os.path.exists(fn3):
            logging.info(f'Bio-Star file for BMRB entry {id_} not found in directory {self.entry_path}')
            # try to find str file in bmrb_dir
            self.entry_path = bmrb_dir
            fn3 = os.path.join(bmrb_dir, f"bmr{id_}_3.str")
            if not os.path.exists(fn3):
                logging.error(f'Bio-Star file for BMRB entry {id_} not found, file {fn3} does not exist')
                raise ValueError

        self.source = fn3
        entry = pynmrstar.Entry.from_file(fn3)
        # entry info
        entry_information = entry.get_saveframes_by_category('entry_information')
        if entry_information:
            self.type = get_tag_vals(entry_information[0], '_Entry.Type', indices=0)
            self.submission_date = get_tag_vals(entry_information[0], '_Entry.Submission_date', indices=0)
            self.nmr_star_version = get_tag_vals(entry_information[0], '_Entry.NMR_STAR_version', indices=0)
            self.original_nmr_star_version = get_tag_vals(entry_information[0], '_Entry.Original_NMR_STAR_version', indices=0)
            self.exp_method = get_tag_vals(entry_information[0], '_Entry.Experimental_method', indices=0)
            self.exp_method_subtype = get_tag_vals(entry_information[0], '_Entry.Experimental_method_subtype', indices=0)
            self.struct_keywords = get_tag_vals(entry_information[0], '_Struct_keywords.Keywords', default=[])
            # database links
            self.db_links = list(zip(
                get_tag_vals(entry_information[0], '_Related_entries.Database_name', default=[]),
                get_tag_vals(entry_information[0], '_Related_entries.Database_accession_code', default=[]),
                get_tag_vals(entry_information[0], '_Related_entries.Relationship', default=[])
                ))

        # citation info
        citations = entry.get_saveframes_by_category('citations')
        if citations:
            self.citation_title = get_tag_vals(citations[0], '_Citation.Title', indices=0)
            self.citation_journal = get_tag_vals(citations[0], '_Citation.Journal_abbrev', indices=0)
            self.citation_PubMed_ID = get_tag_vals(citations[0], '_Citation.PubMed_ID', indices=0)
            self.citation_DOI = get_tag_vals(citations[0], '_Citation.DOI', indices=0)
            self.citation_keywords = get_tag_vals(citations[0], '_Citation_keyword.Keyword')

        # Assembly info
        entry_assemblies = entry.get_saveframes_by_category('assembly')
        if len(entry_assemblies) == 0:
            logging.error(f'BMRB entry {id_} contains no assembly information')
            #raise ValueError
        self.assemblies = [Assembly(sf) for sf in entry_assemblies]
        if not len([a.id for a in self.assemblies]) == len({a.id for a in self.assemblies}):
            logging.error("entry contains assemblies with non-unique ID")
            raise ValueError
        self.assemblies = {a.id:a for a in self.assemblies}
        
        entry_entities = entry.get_saveframes_by_category('entity')
        if len(entry_entities) == 0:
            logging.error(f'BMRB entry {id_} contains no entity information')
            #raise ValueError
        self.entities = [Entity(sf) for sf in entry_entities]
        if not len([e.id for e in self.entities]) == len({e.id for e in self.entities}):
            logging.error("entry contains entities with non-unique ID")
            raise ValueError
        self.entities = {e.id:e for e in self.entities}

        entry_samples = entry.get_saveframes_by_category('sample')
        if len(entry_samples) == 0:
            logging.warning(f'BMRB entry {id_} contains no sample information')
        else:
            self.samples = [Sample(sf) for sf in entry_samples]
            if not len([s.id for s in self.samples]) == len({s.id for s in self.samples}):
                logging.error("entry contains samples with non-unique ID")
                raise ValueError
            self.samples = {s.id:s for s in self.samples}

        entry_conditions = entry.get_saveframes_by_category('sample_conditions')
        if len(entry_conditions) == 0:
            logging.warning(f'BMRB entry {id_} contains no sample condition information')
        else:
            self.conditions = [SampleConditions(sf) for sf in entry_conditions]
            if not len([a.id for a in self.conditions]) == len({a.id for a in self.conditions}):
                logging.error("entry contains conditions with non-unique ID")
                raise ValueError
            self.conditions = {a.id:a for a in self.conditions}

        entry_shift_tables = entry.get_saveframes_by_category('assigned_chemical_shifts')
        if len(entry_shift_tables) == 0:
            logging.error(f'BMRB entry {id_} contains no chemical shift information')
            #raise ValueError
        self.shift_tables = [ShiftTable(sf) for sf in entry_shift_tables]
        if not len([s.id for s in self.shift_tables]) == len({s.id for s in self.shift_tables}):
            logging.error("entry contains shift tables with non-unique ID")
            raise ValueError
        self.shift_tables = {s.id:s for s in self.shift_tables}
    
    def get_peptide_shifts(self):
        peptide_shifts = {}
        for stID,st in self.shift_tables.items():
            condID = st.conditions
            if condID is None or condID not in self.conditions:
                logging.error(f'skipping shift table {stID} due to missing conditions entry: {condID}')
                continue
            for (entity_assemID,entityID),shifts in st.shifts.items():
                # check if there is ambiguity if the entityID tag in matching assemblies is not tested (might be empty)
                assemID = [(self.assemblies[assem].id, e[1]) for assem in self.assemblies for e in self.assemblies[assem].entities if e[0] == entity_assemID and e[1] in ['', None, entityID]]
                if len(assemID) > 1:
                    # if so, enforce same entityID
                    # should be extremely rare, only few entries have multiple assemblies...
                    logging.info(f"ambiguity with respect to correct assemly, enforcing matching, non-empty entityID")
                    assemID = [(self.assemblies[assem].id, e[1]) for assem in self.assemblies for e in self.assemblies[assem].entities if e[0] == entity_assemID and e[1] == entityID]
                    if len(assemID) > 1:
                        logging.error(f'skipping shifts for entity {entityID} due to ambiguity with respect to correct assemly: {assemID}')
                        continue
                if len(assemID) == 0:
                    logging.error(f"skipping shift table {stID}, could not associate it with any assembly")
                    continue
                assemID, entityID_ = assemID[0]
                if entityID_ != entityID:
                    logging.warning(f'Unambiguously using assembly {assemID} with empty entityID tag for assembly entity {entity_assemID}, assuming it is {entityID}')
                
                if entity_assemID is None or entity_assemID not in [e[0] for assem in self.assemblies for e in self.assemblies[assem].entities]: #self.assemblies:
                    logging.error(f'skipping shifts for entity {entityID} due to missing assembly entity entry: {entity_assemID}')
                    continue
                if entityID is None or entityID not in self.entities:
                    logging.error(f'skipping shifts for assembly {entity_assemID} due to missing entity entry: {entityID}')
                    continue
                entity = self.entities[entityID]
                if not entity.type:
                    logging.error(f'skipping shifts for assembly {entity_assemID} due to missing entity type: {entityID}')
                    continue
                if entity.type == 'polymer':
                    if not entity.polymer_type:
                        logging.error(f'skipping shifts for assembly {entity_assemID} due to missing polymer type for entity: {entityID}')
                        continue
                    if entity.polymer_type == 'polypeptide(L)':
                        peptide_shifts[(stID, condID, assemID, entity_assemID, entityID)] = shifts
        return peptide_shifts

    
    def __str__(self):
        def pplist(l):
            if len(l) == 0 : return "[]"
            elif len(l) == 1 : return f"[{str(l[0])}]"
            else: return "[\n    " + "\n    ".join([str(e) for e in l]) + "\n]"
        
        s = f"bmr{self.id}:\n" + "\n  ".join([
            f"id = {self.id}",
            f"citation_title = {self.citation_title}",
            f"citation_journal = {self.citation_journal}",
            f"citation_PubMed_ID = {self.citation_PubMed_ID}",
            f"citation_DOI = {self.citation_DOI}",
            f"citation_keywords = {self.citation_keywords}",
            f"assemblies = {pplist(list(self.assemblies.values()))}",
            f"entities = {pplist(list(self.entities.values()))}",
            f"samples = {pplist(list(self.samples.values()))}",
            f"conditions = {pplist(list(self.conditions.values()))}",
            f"shifts = {pplist(list(self.shift_tables.values()))}",
        ])
        return s
    
    def __repr__(self):
        return f'<bmr{self.id}>'

def get_valid_bbshifts(shifts, seq, filter_amb=True, max_err=1.3):
    bb_atm_ids = ['C','CA','CB','HA','H','N','HB']
    bbshifts = {}
    # 0: '_Atom_chem_shift.Entity_assembly_ID'
    # 1: '_Atom_chem_shift.Entity_ID'
    # 2: '_Atom_chem_shift.Seq_ID'
    # 3: '_Atom_chem_shift.Comp_ID'
    # 4: '_Atom_chem_shift.Atom_ID'
    # 5: '_Atom_chem_shift.Atom_type'
    # 6: '_Atom_chem_shift.Val'
    # 7: '_Atom_chem_shift.Val_err'
    # 8: '_Atom_chem_shift.Ambiguity_code'
    for (_,_,pos1,aa3,atm_id,atm_type,val,err,ambc) in shifts:
        pos0 = int(pos1) - 1
        if pos0 >= len(seq):
            logging.error(f'shift array sequence longer than polymer sequence')
            return
        if aa3 in aa3to1:
            # covert to floats
            try:
                val = float(val)
            except:
                logging.warning(f'skipping shift value of atom_id {atm_id} at 0-based position {pos0}, conversion failed: {val}')
                continue
            try:
                err = float(err)
            except:
                logging.debug(f'setting default for shift error value of atom_id {atm_id} at 0-based position {pos0}, conversion failed: {err}')
                err = np.nan #0. # TODO: correct default value?
            if seq[pos0] != aa3to1[aa3]:
                logging.error(f'canonical amino acid mismatch at 0-based position {pos0}')
                return
            if max_err is not None:
                if np.isnan(err):
                    pass
                elif err <= max_err:
                    pass
                else:
                    continue
            if filter_amb:
                if ambc not in ['1', '2', '', '.']:
                    continue
            if atm_id in bb_atm_ids:
                if pos0 not in bbshifts:
                    bbshifts[pos0] = {}
                if atm_id in bbshifts[pos0]:
                    if bbshifts[pos0][atm_id] != (val, err):
                        logging.error(f'multiple different shifts found for atom_id {atm_id} at 0-based position {pos0}')
                        return
                    else:
                        logging.warning(f'multiple identical shifts found for atom_id {atm_id} at 0-based position {pos0}')
                bbshifts[pos0][atm_id] = (val, err)
            elif aa3 == 'GLY' and atm_id in ['HA2', 'HA3']:
                if pos0 not in bbshifts:
                    bbshifts[pos0] = {}
                if 'HA' not in bbshifts[pos0]:
                    bbshifts[pos0]['HA'] = {}
                if atm_id in bbshifts[pos0]['HA']:
                    if bbshifts[pos0]['HA'][atm_id] != (val, err):
                        logging.error(f'multiple different shifts found for atom_id {atm_id} at 0-based position {pos0}')
                        return
                    else:
                        logging.warning(f'multiple identical shifts found for atom_id {atm_id} at 0-based position {pos0}')
                bbshifts[pos0]['HA'][atm_id] = (val, err)
            elif (aa3 != 'ALA' and atm_id in ['HB2', 'HB3']) or\
                 (aa3 == 'ALA' and atm_id in ['HB1', 'HB2', 'HB3']):
                if pos0 not in bbshifts:
                    bbshifts[pos0] = {}
                if 'HB' not in bbshifts[pos0]:
                    bbshifts[pos0]['HB'] = {}
                if atm_id in bbshifts[pos0]['HB']:
                    if bbshifts[pos0]['HB'][atm_id] != (val, err):
                        logging.error(f'multiple different shifts found for atom_id {atm_id} at 0-based position {pos0}')
                        return
                    else:
                        logging.warning(f'multiple identical shifts found for atom_id {atm_id} at 0-based position {pos0}')
                bbshifts[pos0]['HB'][atm_id] = (val, err)
    for pos0 in bbshifts:
        for atm_id in bbshifts[pos0]:
            if type(bbshifts[pos0][atm_id]) == dict:
                vals = [v[0] for v in bbshifts[pos0][atm_id].values()]
                errs = [v[0] for v in bbshifts[pos0][atm_id].values()]
                bbshifts[pos0][atm_id] = (np.mean(vals), np.mean(errs))
    bbshifts_arr = np.zeros(shape=(len(seq), len(bb_atm_ids)))
    bbshifts_mask = np.full(shape=(len(seq), len(bb_atm_ids)), fill_value=False)
    for i in range(len(seq)):
        for j,atm_id in enumerate(bb_atm_ids):
            if i in bbshifts:
                if atm_id in bbshifts[i]:
                    bbshifts_arr[i,j] = bbshifts[i][atm_id][0]
                    bbshifts_mask[i,j] = True
    return bbshifts, bbshifts_arr, bbshifts_mask