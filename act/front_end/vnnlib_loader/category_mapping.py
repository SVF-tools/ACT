#!/usr/bin/env python3
"""
VNNLIB Category Mapping for ACT Verification Framework.

Provides mapping between VNNLIB benchmark categories and their properties.
Similar to TorchVision's data_model_mapping but for VNNLIB benchmarks.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from typing import Dict, List

from act.util.format_utils import rule

# VNN-COMP 2026 benchmarks: https://github.com/VNN-COMP/vnncomp2026_benchmarks
# Each registry key equals the repo folder name; the loader resolves instances
# under the "<name>/2.0/" version subdirectory by default (override per entry
# with a "version" field).
CATEGORY_MAPPING: Dict[str, Dict] = {
    "acasxu_2023": {
        "type": "collision_avoidance",
        "description": "ACAS Xu collision avoidance neural networks",
        "models": "45 fully-connected networks (5 layers, ReLU)",
        "properties": "Safety properties for aircraft collision avoidance",
        "input_dim": 5,
        "output_dim": 5,
        "year": 2023,
    },
    "cctsdb_yolo_2023": {
        "type": "object_detection",
        "description": "YOLO networks for Chinese traffic sign detection",
        "models": "YOLO-based convolutional networks",
        "properties": "Robustness properties for traffic sign detection",
        "input_dim": "3×640×640",
        "output_dim": "Variable",
        "year": 2023,
    },
    "cersyve": {
        "type": "image_classification",
        "description": "Certified robustness for image classification",
        "models": "Various CNN architectures",
        "properties": "L-inf perturbation robustness",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "cgan_2023": {
        "type": "generative",
        "description": "Conditional GANs verification",
        "models": "Generator and discriminator networks",
        "properties": "GAN stability and output properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2023,
    },
    "cifar100_2024": {
        "type": "image_classification",
        "description": "CIFAR-100 classification networks",
        "models": "ResNet, VGG, and custom CNNs",
        "properties": "Adversarial robustness (L-inf perturbations)",
        "input_dim": "3×32×32",
        "output_dim": 100,
        "year": 2024,
    },
    "collins_aerospace_benchmark": {
        "type": "aerospace",
        "description": "Collins Aerospace verification benchmarks",
        "models": "Industrial aerospace control networks",
        "properties": "Safety-critical system properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "collins_rul_cnn_2022": {
        "type": "remaining_useful_life",
        "description": "Remaining Useful Life prediction with CNNs",
        "models": "CNN-based RUL prediction networks",
        "properties": "Prediction reliability properties",
        "input_dim": "Time series",
        "output_dim": 1,
        "year": 2022,
    },
    "cora_2024": {
        "type": "reachability",
        "description": "CORA reachability analysis benchmarks",
        "models": "Control systems with neural networks",
        "properties": "Reachability and safety properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "dist_shift_2023": {
        "type": "distribution_shift",
        "description": "Distribution shift robustness",
        "models": "Classification networks under distribution shift",
        "properties": "Robustness under input distribution changes",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2023,
    },
    "linearizenn_2024": {
        "type": "linearization",
        "description": "Neural network linearization verification",
        "models": "Networks with piecewise linear activations",
        "properties": "Linearization accuracy properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "lsnc_relu": {
        "type": "control",
        "description": "Learning-enabled state-space controllers",
        "models": "ReLU networks for control systems",
        "properties": "Closed-loop stability properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "malbeware": {
        "type": "malware_detection",
        "description": "Malware detection neural networks",
        "models": "Feed-forward networks for malware classification",
        "properties": "Evasion attack robustness",
        "input_dim": "Variable",
        "output_dim": "Binary",
        "year": 2024,
    },
    "metaroom_2023": {
        "type": "3d_reconstruction",
        "description": "MetaRoom 3D reconstruction networks",
        "models": "Neural networks for 3D scene understanding",
        "properties": "Geometric consistency properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2023,
    },
    "ml4acopf_2024": {
        "type": "power_systems",
        "description": "ML for AC Optimal Power Flow",
        "models": "Neural networks for power grid optimization",
        "properties": "Physical constraint satisfaction (uses sin/cos)",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "nn4sys": {
        "type": "systems",
        "description": "Neural networks for system control and modeling",
        "models": "Control and system identification networks",
        "properties": "System stability and safety properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "relusplitter": {
        "type": "verification_algorithm",
        "description": "ReLUSplitter algorithm test cases",
        "models": "Various ReLU networks",
        "properties": "Algorithm performance benchmarks",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "safenlp_2024": {
        "type": "nlp",
        "description": "Sentence classification robustness on sentence embeddings (Medical safety, R-U-A-Robot)",
        "models": "30-128-2 ReLU MLPs on 30-d PCA-reduced all-MiniLM-L6-v2 sentence embeddings",
        "properties": "Every embedding in a box spanned by perturbed versions of a positive sentence keeps class 0",
        "input_dim": 30,
        "output_dim": 2,
        "year": 2024,
    },
    "sat_relu": {
        "type": "sat_encoding",
        "description": "SAT-based ReLU network verification",
        "models": "ReLU networks for SAT encoding studies",
        "properties": "SAT solver performance benchmarks",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "soundnessbench": {
        "type": "soundness_testing",
        "description": "Verifier soundness testing benchmarks",
        "models": "Networks with known properties",
        "properties": "Ground truth for verifier validation",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "test": {
        "type": "testing",
        "description": "Test cases for verification tools",
        "models": "Small test networks",
        "properties": "Simple test properties",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2024,
    },
    "tinyimagenet_2024": {
        "type": "image_classification",
        "description": "TinyImageNet classification networks",
        "models": "ResNet and VGG variants",
        "properties": "Adversarial robustness (L-inf perturbations)",
        "input_dim": "3×64×64",
        "output_dim": 200,
        "year": 2024,
    },
    "tllverifybench_2023": {
        "type": "transfer_learning",
        "description": "Transfer learning verification benchmarks",
        "models": "Fine-tuned pre-trained networks",
        "properties": "Transfer learning robustness",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2023,
    },
    "traffic_signs_recognition_2023": {
        "type": "traffic_sign_recognition",
        "description": "Traffic sign recognition networks",
        "models": "CNNs for traffic sign classification",
        "properties": "Adversarial robustness for safety-critical applications",
        "input_dim": "3×32×32",
        "output_dim": "Variable",
        "year": 2023,
    },
    "vggnet16_2022": {
        "type": "image_classification",
        "description": "VGG-16 networks verification",
        "models": "VGG-16 architecture variants",
        "properties": "Adversarial robustness",
        "input_dim": "3×224×224",
        "output_dim": 1000,
        "year": 2022,
    },
    "vit_2023": {
        "type": "vision_transformer",
        "description": "Vision Transformer verification",
        "models": "ViT architectures",
        "properties": "Transformer robustness properties",
        "input_dim": "3×224×224",
        "output_dim": "Variable",
        "year": 2023,
    },
    "yolo_2023": {
        "type": "object_detection",
        "description": "YOLO object detection networks",
        "models": "YOLOv3, YOLOv5 variants",
        "properties": "Robustness for object detection",
        "input_dim": "3×640×640",
        "output_dim": "Variable",
        "year": 2023,
    },
    # --- VNN-COMP 2026 additions (new benchmarks in the 2026 suite) ---
    "adaptive_cruise_control_non_linear_2026": {
        "type": "control",
        "description": "Adaptive cruise control with a non-linear plant model",
        "models": "Fully-connected ReLU controllers (Gemm/ReLU)",
        "properties": "Closed-loop safety under non-linear vehicle dynamics",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2026,
    },
    "cgan2026": {
        "type": "generative",
        "description": "Conditional GAN generators (2026 edition)",
        "models": "Generators with Conv/ConvTranspose/Upsample/Sigmoid/Tanh",
        "properties": "Input/output epsilon robustness of generated images",
        "input_dim": "Latent vector + class condition",
        "output_dim": "Image (1 or 3 channels, 32×32 / 64×64)",
        "year": 2026,
    },
    "challenging_certified_training_2026": {
        "type": "image_classification",
        "description": "Challenging instances for certified-training networks",
        "models": "CNNs (Conv/BatchNorm/Gemm/ReLU), including large cnn7",
        "properties": "L-inf adversarial robustness",
        "input_dim": "3×H×W",
        "output_dim": "Variable",
        "year": 2026,
    },
    "isomorphic_acasxu_2026": {
        "type": "equivalence",
        "description": "ACAS Xu network-equivalence (isomorphism) benchmark",
        "models": "Pairs of original vs perturbed ACAS Xu networks (f vs g)",
        "properties": "Functional equivalence between two networks (dual-model instances.csv)",
        "input_dim": 5,
        "output_dim": 5,
        "year": 2026,
    },
    "monotonic_acasxu_2026": {
        "type": "monotonicity",
        "description": "ACAS Xu monotonicity benchmark",
        "models": "ACAS Xu fully-connected ReLU networks",
        "properties": "Output monotonicity with respect to selected inputs",
        "input_dim": 5,
        "output_dim": 5,
        "year": 2026,
    },
    "relusplitter_2026": {
        "type": "verification_algorithm",
        "description": "ReLUSplitter benchmark (2026 edition)",
        "models": "ReLU networks (Gemm/ReLU) with split/unsplit variants",
        "properties": "Robustness equivalence across ReLU-split transformations",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2026,
    },
    "smart_turn_multimodal_2026": {
        "type": "multimodal",
        "description": "Smart-turn multimodal (audio) turn-taking model",
        "models": "Quantized transformer (Conv/Softmax/Erf/Sqrt/Quantize)",
        "properties": "Robustness of turn-taking prediction",
        "input_dim": "Audio features",
        "output_dim": "Variable",
        "year": 2026,
    },
    "soundnessbench_2026": {
        "type": "soundness_testing",
        "description": "Verifier soundness testing benchmark (2026 edition)",
        "models": "Networks with known properties (incl. residual model)",
        "properties": "Ground-truth (un)satisfiable instances for verifier validation",
        "input_dim": "Variable",
        "output_dim": "Variable",
        "year": 2026,
    },
}


def get_category_info(category_name: str) -> Dict:
    """
    Get detailed information for a specific VNNLIB category.
    
    Args:
        category_name: Name of the category (case-sensitive)
        
    Returns:
        Dictionary with category information
        
    Raises:
        ValueError: If category not found
        
    Example:
        >>> info = get_category_info("acasxu_2023")
        >>> print(info['description'])
        ACAS Xu collision avoidance neural networks
    """
    if category_name not in CATEGORY_MAPPING:
        raise ValueError(
            f"Category '{category_name}' not found. "
            f"Use list_categories() to see available categories."
        )
    return CATEGORY_MAPPING[category_name]


def list_categories() -> List[str]:
    """
    List all available VNNLIB categories.
    
    Returns:
        Sorted list of category names
        
    Example:
        >>> categories = list_categories()
        >>> print(f"Total categories: {len(categories)}")
        Total categories: 34
    """
    return sorted(CATEGORY_MAPPING.keys())


def list_categories_by_type(category_type: str) -> List[str]:
    """
    List all categories of a specific type.
    
    Args:
        category_type: Type of category (e.g., 'image_classification', 'control')
        
    Returns:
        List of category names matching the type
        
    Example:
        >>> image_cats = list_categories_by_type('image_classification')
        >>> print(image_cats)
        ['cersyve', 'challenging_certified_training_2026', 'cifar100_2024', 'tinyimagenet_2024', 'vggnet16_2022']
    """
    return [
        name for name, info in CATEGORY_MAPPING.items()
        if info['type'] == category_type
    ]


def get_all_types() -> List[str]:
    """
    Get all unique category types.
    
    Returns:
        Sorted list of category types
        
    Example:
        >>> types = get_all_types()
        >>> print(types)
        ['3d_reconstruction', 'aerospace', 'collision_avoidance', ...]
    """
    types = {info['type'] for info in CATEGORY_MAPPING.values()}
    return sorted(types)


def search_categories(query: str) -> List[str]:
    """
    Search for categories by name or description (case-insensitive).
    
    Args:
        query: Search term
        
    Returns:
        List of matching category names
        
    Example:
        >>> matches = search_categories('yolo')
        >>> print(matches)
        ['cctsdb_yolo_2023', 'yolo_2023']
    """
    query_lower = query.lower()
    matches = []
    
    for name, info in CATEGORY_MAPPING.items():
        if (query_lower in name.lower() or
            query_lower in info['description'].lower() or
            query_lower in info['type'].lower()):
            matches.append(name)
    
    return sorted(matches)


def find_category_name(query: str) -> str:
    """
    Find exact category name with case-insensitive matching.
    
    Args:
        query: Category name (case-insensitive)
        
    Returns:
        Properly cased category name
        
    Raises:
        ValueError: If category not found
        
    Example:
        >>> name = find_category_name('ACASXU_2023')
        >>> print(name)
        acasxu_2023
    """
    query_lower = query.lower()
    
    for name in CATEGORY_MAPPING.keys():
        if name.lower() == query_lower:
            return name
    
    # If exact match not found, suggest similar categories
    similar = search_categories(query)
    if similar:
        raise ValueError(
            f"Category '{query}' not found. Did you mean one of these?\n" +
            "\n".join(f"  • {cat}" for cat in similar[:5])
        )
    else:
        raise ValueError(
            f"Category '{query}' not found. "
            f"Use list_categories() to see all available categories."
        )


def get_summary_statistics() -> Dict:
    """
    Get summary statistics for all VNNLIB categories.
    
    Returns:
        Dictionary with statistics
        
    Example:
        >>> stats = get_summary_statistics()
        >>> print(f"Total categories: {stats['total_categories']}")
        Total categories: 34
    """
    types = get_all_types()
    type_counts = {t: len(list_categories_by_type(t)) for t in types}
    
    return {
        'total_categories': len(CATEGORY_MAPPING),
        'total_types': len(types),
        'categories_by_type': type_counts,
        'oldest_year': min(info['year'] for info in CATEGORY_MAPPING.values()),
        'newest_year': max(info['year'] for info in CATEGORY_MAPPING.values()),
    }


if __name__ == "__main__":
    # Quick demo
    print(rule())
    print("VNNLIB CATEGORY MAPPING")
    print(rule())
    
    stats = get_summary_statistics()
    print(f"\nTotal Categories: {stats['total_categories']}")
    print(f"Category Types: {stats['total_types']}")
    print(f"Year Range: {stats['oldest_year']}-{stats['newest_year']}")
    
    print("\n" + rule())
    print("CATEGORIES BY TYPE")
    print(rule())
    
    for cat_type in sorted(get_all_types()):
        categories = list_categories_by_type(cat_type)
        print(f"\n{cat_type} ({len(categories)}):")
        for cat in categories:
            print(f"  • {cat}")
