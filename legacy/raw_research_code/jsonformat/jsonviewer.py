import json

def inspect_json_structure(file_path, num_items=5, full_annotations=False):
    """
    JSON 구조를 분석하여 주요 정보를 출력하는 함수.
    Args:
        file_path (str): JSON 파일 경로.
        num_items (int): 탐색할 배열/딕셔너리의 최대 항목 수.
        full_annotations (bool): True로 설정 시, annotations의 모든 키를 탐색.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)  # JSON 파일 전체 로드
    
    def inspect_element(element, depth=0):
        indent = '  ' * depth
        if isinstance(element, dict):
            print(f"{indent}Object with {len(element)} keys")
            for key, value in element.items():
                print(f"{indent}  Key: '{key}', Type: {type(value).__name__}")
                if isinstance(value, (dict, list)):
                    inspect_element(value, depth + 1)
        elif isinstance(element, list):
            print(f"{indent}Array of {len(element)} items")
            for i, item in enumerate(element[:num_items]):
                print(f"{indent}  Item {i+1}, Type: {type(item).__name__}")
                inspect_element(item, depth + 1)
            if len(element) > num_items:
                print(f"{indent}  ...")
        else:
            print(f"{indent}{type(element).__name__}: {element}")
    
    print("Inspecting JSON Structure:")
    inspect_element(data)
    
    # annotations 내 모든 키 탐색 (full_annotations=True 시)
    if full_annotations and "annotations" in data:
        print("\nInspecting annotations in detail:")
        for i, annotation in enumerate(data["annotations"][:num_items]):
            print(f"\nAnnotation {i+1}:")
            for key, value in annotation.items():
                print(f"  Key: '{key}', Type: {type(value).__name__}, Value: {value if isinstance(value, (int, float, str)) else '...'}")

# main 함수 정의
if __name__ == "__main__":
    file_path = "train.json"
    inspect_json_structure(file_path, num_items=1, full_annotations=True)
