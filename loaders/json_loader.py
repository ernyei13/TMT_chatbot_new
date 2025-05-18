import json
from typing import Dict, Any
import os


SYSML_TYPE_MAPPING = {
    '_11_5EAPbeta_be00301_1147424179914_458922_958': 'Block',
    '_17_0_3_17530432_1320213131032_5397_2410': 'Proxy_port',
    '_17_0_3_17530432_1320213059640_259248_2362': 'Interface_block',
    '_12_0_be00301_1164123483951_695645_2041': 'Value_property',
    '_11_5EAPbeta_be00301_1147421401018_777370_546': 'Rational',
    '_11_5EAPbeta_be00301_1147430239267_179145_1189': 'ValueType',
    '_11_5EAPbeta_be00301_1147873190330_159934_2220': 'Requirement',
    '_17_0_2_3_b4c02e1_1401221134741_575510_101826' : 'Allocate',
    '_9_0_be00301_1108044563999_784946_1' : 'Sequencer',   
    '_17_0_1_232f03dc_1325612611695_581988_21583' : 'View',
    '_11_5EAPbeta_be00301_1147420760998_43940_227' : 'View',
    '_17_0_2_3_b4c02e1_1380809976981_576006_43117' : 'View',
    '_17_0_1_232f03dc_1325612611695_581988_21583': 'View',
    '_18_0_2_876026b_1427584914110_476375_136242' : 'ESW Component',
    '_17_0_5_407019f_1350499084166_247295_11961' : 'Volume',
    '_12_1_8740266_1172578094250_807075_1551' : 'Hardware Component',
    '_12_1_8740266_1172578094265_898135_1579' : 'Subsystem',
    '_12_1_8740266_1172578094250_839556_1553' : 'Environmental Effect',
    '_17_0_2_3_b4c02e1_1376582877372_304977_35082' : 'Reviewer',
    '_9_0_be00301_1108044380615_150487_0': 'Diagram Info',
    '_12_1_8740266_1173775032859_804702_281' : 'Diagram Specification',
    '_17_0_2_3_897027c_1377650427029_161201_39966' : 'TMT Requirement',
    '_17_0_3_85f027d_1363320204928_98443_3013': 'HierarchyElement',
    '_18_0_2_b4c02e1_1422576771648_76971_83985': 'Object Properties',
    '_17_0_2_3_ff3038a_1383749269646_31940_44489' : 'HierarchyElement',
}

class Element():
    json_data = None

    def __init__(self, json_data):
        self.json_data = json_data
        self.applied_stereotype_ids = json_data.get("_appliedStereotypeIds", [])
        self.documentation = json_data.get("documentation", "")
        self.type = json_data.get("type", "")
        self.id = json_data.get("id", "")
        self.md_extensions_ids = json_data.get("mdExtensionsIds", [])
        self.owner_id = json_data.get("ownerId", "")
        self.sync_element_id = json_data.get("syncElementId", None)
        self.applied_stereotype_instance_id = json_data.get("appliedStereotypeInstanceId", None)
        self.client_dependency_ids = json_data.get("clientDependencyIds", [])
        self.supplier_dependency_ids = json_data.get("supplierDependencyIds", [])
        self.name = json_data.get("name", "")
        self.name_expression = json_data.get("nameExpression", None)
        self.visibility = json_data.get("visibility", None)
        self.template_parameter_id = json_data.get("templateParameterId", None)
        self.deployment_ids = json_data.get("deploymentIds", [])
        self.slot_ids = json_data.get("slotIds", [])
        self.specification = json_data.get("specification", None)
        self.classifier_ids = json_data.get("classifierIds", [])
        self.stereotyped_element_id = json_data.get("stereotypedElementId", None)
        self.in_ref_ids = json_data.get("_inRefIds", [])
        self.elastic_id = json_data.get("_elasticId", "")
        self.ref_id = json_data.get("_refId", "")
        self.modifier = json_data.get("_modifier", "")
        self.modified = json_data.get("_modified", "")
        self.created = json_data.get("_created", "")
        self.creator = json_data.get("_creator", "")
        self.project_id = json_data.get("_projectId", "")
        self.commit_id = json_data.get("_commitId", "")
        self.editable = json_data.get("_editable", False)
        self.sysml_type = []

        self.slots = []
        self.children = []
        self.owner = None

    def get_id(self):
        return self.json_data["id"]
    
    def get_name(self):
        try:
            return self.json_data["name"]
        except:
            return self.json_data["id"]
    
    def get_basic_info(self, max_depth = 0) -> Dict[str, Any]:
        """
        Returns a dictionary containing the basic requested information for the element.
        """
        entry = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "sysml_type": self.sysml_type,
            "slots": [s.get_basic_info(max_depth - 1) for s in self.slots] if max_depth > 0 else [s.get_name() for s in self.slots] ,
            "children": [c.get_basic_info(max_depth - 1) for c in self.children] if max_depth > 0 else [s.get_name() for s in self.children],
            "owner": self.owner.get_basic_info(max_depth - 1) if self.owner is not None and max_depth > 0 else (self.owner.get_name() if self.owner is not None else None)
        }
        # Add documentation only if it exists and is not just whitespace
        if self.documentation and self.documentation.strip():
            entry["documentation"] = self.documentation
        return entry

    def serialize(self, max_depth=0):
        return {
            "applied_stereotype_ids": self.applied_stereotype_ids,
            "documentation": self.documentation,
            "type": self.type,
            "id": self.id,
            "md_extensions_ids": self.md_extensions_ids,
            "owner_id": self.owner_id,
            "sync_element_id": self.sync_element_id,
            "applied_stereotype_instance_id": self.applied_stereotype_instance_id,
            "client_dependency_ids": self.client_dependency_ids,
            "supplier_dependency_ids": self.supplier_dependency_ids,
            "name": self.name,
            "name_expression": self.name_expression,
            "visibility": self.visibility,
            "template_parameter_id": self.template_parameter_id,
            "deployment_ids": self.deployment_ids,
            "slot_ids": self.slot_ids,
            "specification": self.specification,
            "classifier_ids": self.classifier_ids,
            "stereotyped_element_id": self.stereotyped_element_id,
            "in_ref_ids": self.in_ref_ids,
            "elastic_id": self.elastic_id,
            "ref_id": self.ref_id,
            "modifier": self.modifier,
            "modified": self.modified,
            "created": self.created,
            "creator": self.creator,
            "project_id": self.project_id,
            "commit_id": self.commit_id,
            "editable": self.editable,
            "sysml_type": self.sysml_type,
            "slots": [s.serialize(max_depth - 1) for s in self.slots] if max_depth > 0 else [s.get_id() for s in self.slots] ,
            "children": [c.serialize(max_depth - 1) for c in self.children] if max_depth > 0 else [s.get_id() for s in self.children],
            "owner": self.owner.serialize(max_depth - 1) if self.owner is not None and max_depth > 0 else (self.owner.get_id() if self.owner is not None else None)
        }


def load_elements():
    elements = {}
    # Load JSON and print structure

    # Use absolute path based on current file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    response_path = os.path.join(current_dir, "response.json")

    try:
        with open(response_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise Exception(f"❌ Could not find 'elements.json' at {response_path}")
    except json.JSONDecodeError:
        raise Exception("❌ 'elements.json' exists but contains invalid JSON.")
    
    #populate elements
    for element_json in data.get("elements", []):
        e = Element(element_json)
        elements[e.get_id()] = e
    
    # populate slots and children
    for e in elements.values():
        if e.owner_id != '':
            try:
                e.owner = elements[e.owner_id]
                e.owner.children.append(e)
            except:
                continue
        else: 
            continue

    for e in elements.values():
        for slot_id in e.slot_ids:
            try:
                connected_id = slot_id.split("-slot-")[1]
                e.slots.append(elements[connected_id])
            except:
                continue

    # fill sysml_type
    for e in elements.values():
        for stereo in e.applied_stereotype_ids:
            if stereo in SYSML_TYPE_MAPPING:
                e.sysml_type.append(SYSML_TYPE_MAPPING[stereo])
    print(f"Loaded {len(elements)} elements from JSON.")
    return elements

