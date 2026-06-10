import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2
from typing import Dict, Any, Optional, List

def get_kubric_seg_frame_as_obj_index(seg_frame, seg_colors_dict, tolerance: int = 10):
    """Convert segmentation frame to object indices.

    The raw segmentation videos are typically compressed, so the exact RGB values
    stored in *seg_frame* rarely match the values listed in *seg_colors_dict*
    pixel-perfectly.  Instead of requiring an exact match we assign each pixel to
    the object whose reference colour is within a small Euclidean distance
    (``tolerance``) in RGB space.  A value of 3-5 usually works well for Kubric
    data; we expose it as an argument so callers can tweak it if necessary.

    Parameters
    ----------
    seg_frame : ndarray, shape (H, W, 3), dtype uint8
        Frame from the segmentation video.
    seg_colors_dict : Dict[str, List[int]]
        Mapping of *object id ➔ RGB colour* coming from metadata.
    tolerance : int, optional
        Maximum allowed colour distance (in 0-255 RGB space) for a pixel to be
        considered a match.  Defaults to ``10``.

    Returns
    -------
    obj_index : ndarray, shape (H, W), dtype int32
        Image where every pixel contains the corresponding *object id* (or 0 for
        background).
    """

    # --- prepare colour / id lookup arrays --------------------------------------------------
    ref_colours = np.array(list(seg_colors_dict.values()), dtype=np.int16)  # (N, 3)
    obj_ids = np.array([int(k) for k in seg_colors_dict.keys()], dtype=np.int32)  # (N,)

    h, w, _ = seg_frame.shape
    seg_flat = seg_frame.reshape(-1, 3).astype(np.int16)  # (H*W, 3)
    
    # --- compute distance from every pixel to every reference colour -----------------------
    # diff  -> (H*W, N, 3)
    diff = seg_flat[:, None, :] - ref_colours[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (H*W, N)
    
    closest_idx = dist.argmin(axis=1)  # (H*W,)
    closest_dist = dist[np.arange(dist.shape[0]), closest_idx]
    
    obj_index_flat = np.full(seg_flat.shape[0], fill_value=1, dtype=np.int32)
    matched_mask = closest_dist <= tolerance
    obj_index_flat[matched_mask] = obj_ids[closest_idx[matched_mask]]
    
    return obj_index_flat.reshape(h, w)


def get_physprop_as_spatial_data(metadata, first_seg_frame, physprop_type='all', physprops_range = None):
    """Create physprop frame similar to kubric_dataset.py"""
    seg_ids = metadata['segmentation_id']
    # Seg ids can be something like [5, 7, 1, 2]
    # We need to map them to [2, 3, 0, 1]
    object_ids = np.zeros_like(seg_ids)
    for i, seg_id in enumerate(seg_ids):
        object_ids[i] = seg_ids.index(seg_id)
    
    mass = metadata['mass']
    bounciness = metadata['restitution']
    static_friction = metadata['friction']
    
    if physprop_type == "mass_only" or physprop_type == "mass":
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
    elif physprop_type == "bounciness_only" or physprop_type == "bounciness":
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=0, dtype=np.float32)
    elif physprop_type == "friction_only" or physprop_type == "friction":
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
    elif physprop_type == "all":
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
        physprop_frame[:, :, 0] = 1
        physprop_frame[:, :, 1] = 0
        physprop_frame[:, :, 2] = 1
    # Check if there's a substring "friction" in physprop_type
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
        physprop_frame[:, :, 0] = 0
        physprop_frame[:, :, 1] = 1
        physprop_frame[:, :, 2] = 1
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")
    
    for k, (obj_id, seg_id) in enumerate(zip(object_ids, seg_ids)):
        obj_mask = first_seg_frame == seg_id

        static_friction_val = np.clip(static_friction[obj_id], 0.0, 1.0)
        bounciness_val = np.clip(bounciness[obj_id], 0.0, 1.0)
        mass_val = np.clip(np.log(mass[obj_id] + 1) / np.log(100), 0.0, 1.0)

        if physprops_range is not None and obj_id > 0:
            if physprops_range["friction"] == "low":
                static_friction_val = 0.2 + np.random.rand() * 0.1
            elif physprops_range["friction"] == "high":
                static_friction_val = 0.90 + np.random.rand() * 0.09
            if physprops_range["bounciness"] == "low":
                bounciness_val = 0.05 + np.random.rand() * 0.05
            elif physprops_range["bounciness"] == "high":
                bounciness_val = 0.90 + np.random.rand() * 0.09
            if physprops_range["mass"] == "low":
                mass_val = 0.05 + np.random.rand() * 0.05
            elif physprops_range["mass"] == "high":
                mass_val = 0.90 + np.random.rand() * 0.09

        if physprop_type == 'mass_only' or physprop_type == 'mass':
            physprop_frame[obj_mask, 0] = mass_val
            physprop_frame[obj_mask, 1] = mass_val
            physprop_frame[obj_mask, 2] = mass_val
        elif physprop_type == 'friction_only' or physprop_type == 'friction':
            physprop_frame[obj_mask, 0] = static_friction_val
            physprop_frame[obj_mask, 1] = static_friction_val
            physprop_frame[obj_mask, 2] = static_friction_val
        elif physprop_type == 'bounciness_only' or physprop_type == 'bounciness':
            physprop_frame[obj_mask, 0] = bounciness_val
            physprop_frame[obj_mask, 1] = bounciness_val
            physprop_frame[obj_mask, 2] = bounciness_val
        elif physprop_type == 'all':
            physprop_frame[obj_mask, 0] = mass_val
            physprop_frame[obj_mask, 1] = bounciness_val
            physprop_frame[obj_mask, 2] = static_friction_val
        elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
            physprop_frame[obj_mask, 0] = bounciness_val
            physprop_frame[obj_mask, 1] = static_friction_val
            physprop_frame[obj_mask, 2] = 1.0
        else:
            raise ValueError(f"Invalid physprop type: {physprop_type}")
    return physprop_frame


def get_physprop_as_fg_bg_vector(metadata, physprop_type='all', physprops_range = None):
    """A physprop Vector where there are only two objects background and foreground. The physprop is in the vector form"""
    seg_ids = metadata['segmentation_id']
    # Seg ids can be something like [5, 7, 1, 2]
    # We need to map them to [2, 3, 0, 1]
    object_ids = np.zeros_like(seg_ids)
    for i, seg_id in enumerate(seg_ids):
        object_ids[i] = seg_ids.index(seg_id)
    
    mass = metadata['mass']
    bounciness = metadata['restitution']
    static_friction = metadata['friction']
    
    if physprop_type == "mass_only" or physprop_type == "mass":
        physprop_vector = np.full((2,), fill_value=1, dtype=np.float32)
    elif physprop_type == "bounciness_only" or physprop_type == "bounciness":
        physprop_vector = np.full((2,), fill_value=0, dtype=np.float32)
    elif physprop_type == "friction_only" or physprop_type == "friction":
        physprop_vector = np.full((2,), fill_value=1, dtype=np.float32)
    elif physprop_type == "all":
        physprop_vector = np.array([1,1,0,0,1,1], dtype=np.float32)
    # Check if there's a substring "friction" in physprop_type
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        physprop_vector = np.array([0,0,1,1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")
    
    # Take the last one and the one before it as the foreground and background
    fg_obj_id = np.max(object_ids)
    bg_obj_id = fg_obj_id - 1

    fg_static_friction_val = np.clip(static_friction[fg_obj_id], 0.0, 1.0)
    fg_bounciness_val = np.clip(bounciness[fg_obj_id], 0.0, 1.0)
    fg_mass_val = np.clip(np.log(mass[fg_obj_id] + 1) / np.log(100), 0.0, 1.0)

    bg_static_friction_val = np.clip(static_friction[bg_obj_id], 0.0, 1.0)
    bg_bounciness_val = np.clip(bounciness[bg_obj_id], 0.0, 1.0)
    bg_mass_val = np.clip(np.log(mass[bg_obj_id] + 1) / np.log(100), 0.0, 1.0)
    

    if physprops_range is not None:
        if physprops_range["friction"] == "low":
            fg_static_friction_val = 0.3 + np.random.rand() * 0.1
            bg_static_friction_val = 0.3 + np.random.rand() * 0.1
        elif physprops_range["friction"] == "high":
            fg_static_friction_val = 0.90 + np.random.rand() * 0.09
            bg_static_friction_val = 0.90 + np.random.rand() * 0.09
        if physprops_range["bounciness"] == "low":
            fg_bounciness_val = 0.05 + np.random.rand() * 0.05
            bg_bounciness_val = 0.05 + np.random.rand() * 0.05
        elif physprops_range["bounciness"] == "high":
            fg_bounciness_val = 0.90 + np.random.rand() * 0.09
            bg_bounciness_val = 0.90 + np.random.rand() * 0.09
        if physprops_range["mass"] == "low":
            fg_mass_val = 0.05 + np.random.rand() * 0.05
            bg_mass_val = 0.05 + np.random.rand() * 0.05
        elif physprops_range["mass"] == "high":
            fg_mass_val = 0.90 + np.random.rand() * 0.09
            bg_mass_val = 0.90 + np.random.rand() * 0.09
    
    if physprop_type == "mass_only" or physprop_type == "mass":
        physprop_vector[0] = fg_mass_val
        physprop_vector[1] = bg_mass_val
    elif physprop_type == "bounciness_only" or physprop_type == "bounciness":
        physprop_vector[0] = fg_bounciness_val
        physprop_vector[1] = bg_bounciness_val
    elif physprop_type == "friction_only" or physprop_type == "friction":
        physprop_vector[0] = fg_static_friction_val
        physprop_vector[1] = bg_static_friction_val
    elif physprop_type == "all":
        physprop_vector[0] = fg_mass_val
        physprop_vector[1] = bg_mass_val
        physprop_vector[2] = fg_bounciness_val
        physprop_vector[3] = bg_bounciness_val
        physprop_vector[4] = fg_static_friction_val
        physprop_vector[5] = bg_static_friction_val
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        physprop_vector[0] = fg_bounciness_val
        physprop_vector[1] = bg_bounciness_val
        physprop_vector[2] = fg_static_friction_val
        physprop_vector[3] = bg_static_friction_val
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")
    
    return physprop_vector

def get_physprop_as_image_blob(metadata, seg_frame, physprop_type='all', physprops_range = None, fg_seg_id=None, bg_seg_id=None, fg_seg_ids=None, return_neg_physprop=False, fg_object_types=None, props_of_interest=None, blob_type="circle", return_physprop_text_labels=False, no_background_condition=False):
    """Here physprop consists of background and foreground in the form of an image. The background image is just an image with the physical property value. 
    While the foreground image is a blob with the physical property value at the location of the object(s). 
    
    The blob type determines the shape of the foreground mask:
    - 'circle': Uses the segmentation frame to calculate object center and draws a circle with radius based on mask area
    - 'convex_hull': Computes the convex hull of the original segmentation mask, preserving more of the object's shape
    - 'ellipse': Fits an ellipse to the segmentation mask, providing a compact smooth representation that captures orientation and aspect ratio
    
    Args:
        metadata: Dictionary containing physical properties and segmentation IDs
        seg_frame: Segmentation frame with object IDs
        physprop_type: Type of physical properties to include ('all', 'mass', 'bounciness', 'friction', etc.)
        physprops_range: Optional range specification for properties
        fg_seg_id: Single foreground segmentation ID (for backward compatibility)
        fg_seg_ids: List of foreground segmentation IDs (new functionality for multiple objects)
        bg_seg_id: Background segmentation ID
        return_neg_physprop: Whether to return negative physical properties
        object_types: List of object types (e.g. ['soft', 'rigid'])
        props_of_interest: Property of interest (e.g. ['mass', 'bounciness', 'friction'])
        blob_type: Type of blob to use ('circle', 'convex_hull', or 'ellipse')
        return_physprop_text_labels: Whether to return physical property text labels
        
    Returns:
        physprop_frame: Image with physical properties as pixel values
        neg_physprop_frame: (optional) Image with negative physical properties
    """
    
    seg_ids = metadata['segmentation_id']
    # Seg ids can be something like [5, 7, 1, 2]
    # We need to map them to [2, 3, 0, 1]
    object_ids = np.zeros_like(seg_ids)
    for i, seg_id in enumerate(seg_ids):
        object_ids[i] = seg_ids.index(seg_id)
    
    raw_mass_default_val = 2.0
    raw_bounciness_default_val = 0.0
    raw_friction_default_val = 1.0
    raw_neo_hookean_mu_default_val = 600.0
    raw_neo_hookean_lambda_default_val = 600.0
    raw_neo_hookean_damping_default_val = 1.0
    raw_force_magnitude_default_val = 1.0
    raw_dir_angle_default_val = 0.0
    if props_of_interest is not None:
        mass = metadata['mass'] if 'mass' in props_of_interest else [raw_mass_default_val] * len(seg_ids)
        bounciness = metadata['restitution'] if 'bounciness' in props_of_interest else [raw_bounciness_default_val] * len(seg_ids)
        static_friction = metadata['friction'] if 'friction' in props_of_interest else [raw_friction_default_val] * len(seg_ids)
        neo_hookean_mu = metadata.get('neo_hookean_mu', None) if 'deformable' in props_of_interest else None
        neo_hookean_lambda = metadata.get('neo_hookean_lambda', None) if 'deformable' in props_of_interest else None
        neo_hookean_damping = metadata.get('neo_hookean_damping', None) if 'deformable' in props_of_interest else None
        force_magnitude = metadata.get('force_magnitude', None) if 'force' in props_of_interest else None
        dir_start_img_coords = metadata.get('dir_start_image_coordinates', None) if 'move_dir' in props_of_interest or 'force' in props_of_interest else None
        dir_end_img_coords = metadata.get('dir_end_image_coordinates', None) if 'move_dir' in props_of_interest or 'force' in props_of_interest else None
        dir_angle = metadata.get('dir_angle', None) if 'move_dir' in props_of_interest or 'force' in props_of_interest else None
    else:
        mass = metadata['mass']
        bounciness = metadata['restitution']
        static_friction = metadata['friction']
        neo_hookean_mu = metadata.get('neo_hookean_mu', None)
        neo_hookean_lambda = metadata.get('neo_hookean_lambda', None)
        neo_hookean_damping = metadata.get('neo_hookean_damping', None)
        force_magnitude = metadata.get('force_magnitude', None)
        dir_start_img_coords = metadata.get('dir_start_image_coordinates', None)
        dir_end_img_coords = metadata.get('dir_end_image_coordinates', None)
        dir_angle = metadata.get('dir_angle', None)
    move_object_seg_id = metadata.get('move_object_seg_id', None)
    force_mag_min = metadata.get('force_magnitude_min', 150)
    force_mag_max = metadata.get('force_magnitude_max', 450)
    neo_hookean_mu_min = metadata.get('neo_hookean_mu_min', 60)
    neo_hookean_mu_max = metadata.get('neo_hookean_mu_max', 600)
    neo_hookean_lambda_min = metadata.get('neo_hookean_lambda_min', 100)
    neo_hookean_lambda_max = metadata.get('neo_hookean_lambda_max', 600)

    # Get image dimensions
    H, W = seg_frame.shape[:2]
    
    if physprop_type == "mass_only" or physprop_type == "mass":
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
    elif physprop_type == "bounciness_only" or physprop_type == "bounciness":
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=0, dtype=np.float32)
    elif physprop_type == "friction_only" or physprop_type == "friction":
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
    elif physprop_type == "all":
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
        physprop_frame[:, :, 0] = 1
        physprop_frame[:, :, 1] = 0
        physprop_frame[:, :, 2] = 1
    # Check if there's a substring "friction" in physprop_type
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
        physprop_frame[:, :, 0] = 0.0
        physprop_frame[:, :, 1] = 1
        physprop_frame[:, :, 2] = 1
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" in physprop_type:
        physprop_frame = np.full((seg_frame.shape[0], seg_frame.shape[1], 3), fill_value=1, dtype=np.float32)
        physprop_frame[:, :, 0] = 0.0
        physprop_frame[:, :, 1] = 1
        physprop_frame[:, :, 2] = 1
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")
    
    if "deformable" in physprop_type:
        physprop_frame2 = np.zeros_like(physprop_frame)
        physprop_frame2[:, :, :3] = 1.0
    if "force" in physprop_type or "move_dir" in physprop_type:
        physprop_frame3 = np.zeros_like(physprop_frame)
    
    if return_neg_physprop:
        neg_physprop_frame = physprop_frame.copy()
        if "deformable" in physprop_type:
            neg_physprop_frame2 = physprop_frame2.copy()
        if "force" in physprop_type or "move_dir" in physprop_type:
            neg_physprop_frame3 = physprop_frame3.copy()

    # Determine foreground object IDs
    if fg_seg_ids is not None:
        # Multiple foreground objects specified
        fg_obj_ids = [seg_ids.index(seg_id) for seg_id in fg_seg_ids if seg_id in seg_ids]
        if len(fg_obj_ids) == 0:
            raise ValueError("None of the specified fg_seg_ids found in segmentation_id")
    elif fg_seg_id is not None:
        # Single foreground object specified (backward compatibility)
        fg_obj_ids = [seg_ids.index(fg_seg_id)]
    else:
        # Default behavior - use max object ID
        fg_obj_ids = [np.max(object_ids)]
    
    if move_object_seg_id is not None:
        move_obj_id = seg_ids.index(move_object_seg_id)
    else:
        move_obj_id = fg_obj_ids[0]

    # Determine background object ID
    if bg_seg_id is not None:
        bg_obj_id = seg_ids.index(bg_seg_id)
    else:
        # Use the first foreground object ID - 1 as default background
        bg_obj_id = fg_obj_ids[0] - 1
    
    # Get physical property values for foreground and background objects
    fg_static_friction_vals = [np.clip(static_friction[obj_id], 0.0, 1.0) for obj_id in fg_obj_ids]
    fg_bounciness_vals = [np.clip(bounciness[obj_id], 0.0, 1.0) for obj_id in fg_obj_ids]
    # fg_mass_vals = [np.clip(mass[obj_id]/raw_mass_default_val, 0.0, 1.0) for obj_id in fg_obj_ids]
    fg_mass_vals = [1.0 for obj_id in fg_obj_ids]
    
    bg_static_friction_val = np.clip(static_friction[bg_obj_id], 0.0, 1.0)
    bg_bounciness_val = np.clip(bounciness[bg_obj_id], 0.0, 1.0)
    bg_mass_val = 1.0
    if "force" in physprop_type or "move_dir" in physprop_type:
        if dir_angle is None and dir_start_img_coords is not None and dir_end_img_coords is not None:
            dir_angle = np.arctan2(dir_end_img_coords[1] - dir_start_img_coords[1], dir_end_img_coords[0] - dir_start_img_coords[0])
            move_dir_sin_theta_val = (np.sin(dir_angle) + 1.0)/2.0
            move_dir_cos_theta_val = (np.cos(dir_angle) + 1.0)/2.0
        elif dir_angle is not None:
            move_dir_sin_theta_val = (np.sin(dir_angle) + 1.0)/2.0
            move_dir_cos_theta_val = (np.cos(dir_angle) + 1.0)/2.0
        else:
            move_dir_sin_theta_val = 0.0
            move_dir_cos_theta_val = 0.0
    if "force" in physprop_type:
        if force_magnitude is not None and not np.isnan(force_magnitude):
            force_magnitude = (force_magnitude - force_mag_min)/(force_mag_max - force_mag_min)    
        fg_force_magnitude_val = force_magnitude if force_magnitude is not None and not np.isnan(force_magnitude) else 0.0
        
    if "deformable" in physprop_type:
        # Fix: Properly handle None values and close list comprehensions
        # Replace None with default values before normalization
        if neo_hookean_mu is not None:
            neo_hookean_mu = [mu_val if mu_val is not None else raw_neo_hookean_mu_default_val for mu_val in neo_hookean_mu]
            neo_hookean_mu = (np.array(neo_hookean_mu) - neo_hookean_mu_min) / (neo_hookean_mu_max - neo_hookean_mu_min)
        else:
            neo_hookean_mu = [1.0 for _ in range(len(mass))]
        if neo_hookean_lambda is not None:
            neo_hookean_lambda = [lambda_val if lambda_val is not None else raw_neo_hookean_lambda_default_val for lambda_val in neo_hookean_lambda]
            neo_hookean_lambda = (np.array(neo_hookean_lambda) - neo_hookean_lambda_min) / (neo_hookean_lambda_max - neo_hookean_lambda_min)
        else:
            neo_hookean_lambda = [1.0 for _ in range(len(mass))]
        if neo_hookean_damping is not None:
            neo_hookean_damping = [damping_val if damping_val is not None else 1.0 for damping_val in neo_hookean_damping]
        else:
            neo_hookean_damping = [1.0 for _ in range(len(mass))]
        neo_hookean_damping = np.array(neo_hookean_damping)
        fg_neo_hookean_mu_vals = [neo_hookean_mu[obj_id] if not np.isnan(neo_hookean_mu[obj_id]) else 1.0 for obj_id in fg_obj_ids]
        bg_neo_hookean_mu_val = neo_hookean_mu[bg_obj_id] if not np.isnan(neo_hookean_mu[bg_obj_id]) else 1.0
        fg_neo_hookean_lambda_vals = [neo_hookean_lambda[obj_id] if not np.isnan(neo_hookean_lambda[obj_id]) else 1.0 for obj_id in fg_obj_ids]
        bg_neo_hookean_lambda_val = neo_hookean_lambda[bg_obj_id] if not np.isnan(neo_hookean_lambda[bg_obj_id]) else 1.0
        fg_neo_hookean_damping_vals = [neo_hookean_damping[obj_id] if not np.isnan(neo_hookean_damping[obj_id]) else 1.0 for obj_id in fg_obj_ids]
        bg_neo_hookean_damping_val = neo_hookean_damping[bg_obj_id] if not np.isnan(neo_hookean_damping[bg_obj_id]) else 1.0
    
    if return_neg_physprop:
        neg_fg_static_friction_vals = fg_static_friction_vals.copy()
        neg_bg_static_friction_val = bg_static_friction_val
        neg_fg_bounciness_vals = fg_bounciness_vals.copy()
        neg_bg_bounciness_val = bg_bounciness_val
        neg_fg_mass_vals = fg_mass_vals.copy()
        neg_bg_mass_val = bg_mass_val
        if "deformable" in physprop_type:
            neg_fg_neo_hookean_mu_vals = fg_neo_hookean_mu_vals.copy()
            neg_bg_neo_hookean_mu_val = bg_neo_hookean_mu_val
            neg_fg_neo_hookean_lambda_vals = fg_neo_hookean_lambda_vals.copy()
            neg_bg_neo_hookean_lambda_val = bg_neo_hookean_lambda_val
            neg_fg_neo_hookean_damping_vals = fg_neo_hookean_damping_vals.copy()
            neg_bg_neo_hookean_damping_val = bg_neo_hookean_damping_val
        if "force" in physprop_type:
            neg_fg_force_sin_theta_vals = move_dir_sin_theta_val.copy()
            neg_fg_force_cos_theta_vals = move_dir_cos_theta_val.copy()
            neg_fg_force_magnitude_val = fg_force_magnitude_val.copy()
    
    # Apply physprops_range if provided
    if physprops_range is not None:
        if physprops_range["friction"] == "low" and "friction" in props_of_interest:
            fg_static_friction_vals = [0.2 + np.random.rand() * 0.1 for _ in fg_obj_ids]
            bg_static_friction_val = 0.2 + np.random.rand() * 0.1
        elif physprops_range["friction"] == "high" and "friction" in props_of_interest:
            fg_static_friction_vals = [0.90 + np.random.rand() * 0.09 for _ in fg_obj_ids]
            bg_static_friction_val = 0.90 + np.random.rand() * 0.09
        elif physprops_range["friction"] == "medium" and "friction" in props_of_interest:
            fg_static_friction_vals = [0.4 + np.random.rand() * 0.1 for _ in fg_obj_ids]
            bg_static_friction_val = 0.5 + np.random.rand() * 0.1
        # Handle bounciness - can be a single value or a list
        bounciness_setting = physprops_range["bounciness"]
        if "bounciness" in props_of_interest:
            # Convert single value to list
            if not isinstance(bounciness_setting, list):
                bounciness_setting = [bounciness_setting] * len(fg_obj_ids)     
            # Process as list
            if no_background_condition:
                bg_bounciness_vals = [0.0 for _ in range(len(fg_obj_ids))]
            if "low" in bounciness_setting or "high" in bounciness_setting or "medium" in bounciness_setting:
                for k, val in enumerate[Any](bounciness_setting):
                    if k < len(fg_bounciness_vals):
                        if val == "low":
                            fg_bounciness_vals[k] = 0.05 + np.random.rand() * 0.1
                            if not no_background_condition:
                                bg_bounciness_val = 0.05 + np.random.rand() * 0.1
                            else:
                                bg_bounciness_vals[k] = 0.05 + np.random.rand() * 0.1
                        elif val == "high":
                            fg_bounciness_vals[k] = 0.90 + np.random.rand() * 0.09
                            if not no_background_condition:
                                bg_bounciness_val = 0.90 + np.random.rand() * 0.09
                            else:
                                bg_bounciness_vals[k] = 0.90 + np.random.rand() * 0.09
                        elif val == "medium":
                            fg_bounciness_vals[k] = 0.5 + np.random.rand() * 0.1
                            if not no_background_condition:
                                bg_bounciness_val = 0.5 + np.random.rand() * 0.1
                            else:
                                bg_bounciness_vals[k] = 0.5 + np.random.rand() * 0.1
        if physprops_range["mass"] == "low":
            fg_mass_vals = [1.0/4.0 for _ in fg_obj_ids]
            bg_mass_val = 1.0
        elif physprops_range["mass"] == "high":
            fg_mass_vals = [3.0/4.0 for _ in fg_obj_ids]
            bg_mass_val = 1.0
        elif physprops_range["mass"] == "medium":
            fg_mass_vals = [2.0/4.0 for _ in fg_obj_ids]
            bg_mass_val = 1.0
        # Handle neo_hookean_mu - can be a single value or a list
        neo_hookean_mu_setting = physprops_range["neo_hookean_mu"]
        if "deformable" in props_of_interest:
            # Convert single value to list
            if not isinstance(neo_hookean_mu_setting, list):
                neo_hookean_mu_setting = [neo_hookean_mu_setting] * len(fg_obj_ids)
            
            # Process as list
            for k, val in enumerate(neo_hookean_mu_setting):
                if k < len(fg_neo_hookean_mu_vals) and val:  # val not empty string
                    if fg_object_types is None or fg_object_types[k] == "soft":
                        if val == "low":
                            fg_neo_hookean_mu_vals[k] = 0.01 + np.random.rand() * 0.08
                        elif val == "high":
                            fg_neo_hookean_mu_vals[k] = 0.80 + np.random.rand() * 0.19
                        elif val == "medium":
                            fg_neo_hookean_mu_vals[k] = 0.3 + np.random.rand() * 0.1
            bg_neo_hookean_mu_val = 1.0
        if physprops_range["neo_hookean_lambda"] == "low" and "deformable" in props_of_interest:
            if fg_object_types is not None:
                for k, _ in enumerate(fg_object_types):
                    if fg_object_types[k] == "soft":
                        fg_neo_hookean_lambda_vals[k] = 0.05 + np.random.rand() * 0.05
            else:
                fg_neo_hookean_lambda_vals = [0.05 + np.random.rand() * 0.05 for _ in fg_obj_ids]
            bg_neo_hookean_lambda_val = 0.05 + np.random.rand() * 0.05
        elif physprops_range["neo_hookean_lambda"] == "high" and "deformable" in props_of_interest:
            if fg_object_types is not None:
                for k, _ in enumerate(fg_object_types):
                    if fg_object_types[k] == "soft":
                        fg_neo_hookean_lambda_vals[k] = 0.90 + np.random.rand() * 0.09
            else:
                fg_neo_hookean_lambda_vals = [0.90 + np.random.rand() * 0.09 for _ in fg_obj_ids]
            bg_neo_hookean_lambda_val = 0.90 + np.random.rand() * 0.09
        elif physprops_range["neo_hookean_lambda"] == "medium" and "deformable" in props_of_interest:
            if fg_object_types is not None:
                for k, _ in enumerate(fg_object_types):
                    if fg_object_types[k] == "soft":
                        fg_neo_hookean_lambda_vals[k] = 0.50 + np.random.rand() * 0.1
            else:
                fg_neo_hookean_lambda_vals = [0.50 + np.random.rand() * 0.1 for _ in fg_obj_ids]
            bg_neo_hookean_lambda_val = 0.50 + np.random.rand() * 0.1
        # Handle neo_hookean_damping - can be a single value or a list
        neo_hookean_damping_setting = physprops_range["neo_hookean_damping"]
        if "deformable" in props_of_interest:
            # Convert single value to list
            if not isinstance(neo_hookean_damping_setting, list):
                neo_hookean_damping_setting = [neo_hookean_damping_setting] * len(fg_obj_ids)
            
            # Process as list
            for k, val in enumerate(neo_hookean_damping_setting):
                if k < len(fg_neo_hookean_damping_vals) and val:  # val not empty string
                    if fg_object_types is None or fg_object_types[k] == "soft":
                        if val == "low":
                            fg_neo_hookean_damping_vals[k] = 0.01 + np.random.rand() * 0.19
                        elif val == "high":
                            fg_neo_hookean_damping_vals[k] = 0.80 + np.random.rand() * 0.09
                        elif val == "medium":
                            fg_neo_hookean_damping_vals[k] = 0.3 + np.random.rand() * 0.1
            bg_neo_hookean_damping_val = 1.0
    if physprops_range is not None and return_neg_physprop:
        if physprops_range["friction"] == "low":
            neg_fg_static_friction_vals = [1.0 - val for val in fg_static_friction_vals]
            neg_bg_static_friction_val = 1.0 - bg_static_friction_val
        elif physprops_range["friction"] == "high":
            neg_fg_static_friction_vals = [1.0 - val for val in fg_static_friction_vals]
            neg_bg_static_friction_val = 1.0 - bg_static_friction_val
        elif physprops_range["friction"] == "medium":
            neg_fg_static_friction_vals = [1.0 - val for val in fg_static_friction_vals]
            neg_bg_static_friction_val = 1.0 - bg_static_friction_val
        if physprops_range["bounciness"] == "low":
            neg_fg_bounciness_vals = [1.0 - val for val in fg_bounciness_vals]
            neg_bg_bounciness_val = 1.0 - bg_bounciness_val
        elif physprops_range["bounciness"] == "high":
            neg_fg_bounciness_vals = [1.0 - val for val in fg_bounciness_vals]
            neg_bg_bounciness_val = 1.0 - bg_bounciness_val
        elif physprops_range["bounciness"] == "medium":
            neg_fg_bounciness_vals = [1.0 - val for val in fg_bounciness_vals]
            neg_bg_bounciness_val = 1.0 - bg_bounciness_val
        if physprops_range["mass"] == "low":
            neg_fg_mass_vals = [1.0 - val for val in fg_mass_vals]
            neg_bg_mass_val = 1.0 - bg_mass_val
        elif physprops_range["mass"] == "high":
            neg_fg_mass_vals = [1.0 - val for val in fg_mass_vals]
            neg_bg_mass_val = 1.0 - bg_mass_val
        elif physprops_range["mass"] == "medium":
            neg_fg_mass_vals = [1.0 - val for val in fg_mass_vals]
            neg_bg_mass_val = 1.0 - bg_mass_val
        if physprops_range["neo_hookean_mu"] == "low":
            neg_fg_neo_hookean_mu_vals = [1.0 - val for val in fg_neo_hookean_mu_vals]
            neg_bg_neo_hookean_mu_val = 1.0 - bg_neo_hookean_mu_val
        elif physprops_range["neo_hookean_mu"] == "high":
            neg_fg_neo_hookean_mu_vals = [1.0 - val for val in fg_neo_hookean_mu_vals]
            neg_bg_neo_hookean_mu_val = 1.0 - bg_neo_hookean_mu_val
        if physprops_range["neo_hookean_lambda"] == "low":
            neg_fg_neo_hookean_lambda_vals = [1.0 - val for val in fg_neo_hookean_lambda_vals]
            neg_bg_neo_hookean_lambda_val = 1.0 - bg_neo_hookean_lambda_val
        elif physprops_range["neo_hookean_lambda"] == "high":
            neg_fg_neo_hookean_lambda_vals = [1.0 - val for val in fg_neo_hookean_lambda_vals]
            neg_bg_neo_hookean_lambda_val = 1.0 - bg_neo_hookean_lambda_val
        if physprops_range["neo_hookean_damping"] == "low":
            neg_fg_neo_hookean_damping_vals = [1.0 - val for val in fg_neo_hookean_damping_vals]
            neg_bg_neo_hookean_damping_val = 1.0 - bg_neo_hookean_damping_val
        elif physprops_range["neo_hookean_damping"] == "high":
            neg_fg_neo_hookean_damping_vals = [1.0 - val for val in fg_neo_hookean_damping_vals]
            neg_bg_neo_hookean_damping_val = 1.0 - bg_neo_hookean_damping_val
    
    # Create masks for all foreground objects
    fg_masks = []
    for i, fg_obj_id in enumerate(fg_obj_ids):
        fg_seg_id = seg_ids[fg_obj_id]
        fg_mask_raw = seg_frame == fg_seg_id
        fg_mask = fg_mask_raw.copy()
        
        if np.any(fg_mask):
            if blob_type == "circle":
                # Calculate object center
                y_coords, x_coords = np.where(fg_mask)
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))

                # Calculate radius based on mask area (equivalent circle radius)
                area = np.sum(fg_mask)
                radius = int(np.sqrt(area / np.pi))

                # Create coordinate grids
                y_grid, x_grid = np.ogrid[:H, :W]

                # Calculate distance from center
                dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

                # Create circular mask
                circle_mask = dist_from_center <= radius
                fg_mask = circle_mask
            elif blob_type == "ellipse":
                # Get the points of the mask
                y_coords, x_coords = np.where(fg_mask)
                
                if len(y_coords) >= 5:  # Need at least 5 points to fit an ellipse
                    # Stack coordinates as (x, y) pairs for fitEllipse
                    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
                    
                    try:
                        # Fit ellipse to the points
                        # fitEllipse returns ((center_x, center_y), (width, height), angle)
                        ellipse = cv2.fitEllipse(points)
                        
                        # Create a new mask with the ellipse (uint8 for cv2)
                        fg_mask_temp = np.zeros((H, W), dtype=np.uint8)
                        
                        # Draw filled ellipse
                        cv2.ellipse(fg_mask_temp, ellipse, 1, -1)
                        
                        # Convert to boolean mask
                        fg_mask = fg_mask_temp.astype(bool)
                    except Exception as e:
                        # If ellipse fitting fails, keep the original mask
                        fg_mask = fg_mask_raw.copy()
                # else: keep the original mask if not enough points
            elif blob_type == "convex_hull":
                # Get the points of the mask
                y_coords, x_coords = np.where(fg_mask)
                
                if len(y_coords) >= 3:  # Need at least 3 points for a convex hull
                    # Check if points are degenerate (collinear or near-collinear)
                    # A convex hull needs points that span 2D space
                    x_range = x_coords.max() - x_coords.min()
                    y_range = y_coords.max() - y_coords.min()
                    
                    # If points are essentially on a line (one dimension has no range),
                    # or the area is too small, fall back to original mask
                    if x_range < 2 or y_range < 2:
                        # Points are degenerate (line or single point), keep original mask
                        fg_mask = fg_mask_raw.copy()
                    else:
                        # Stack coordinates as (x, y) pairs for ConvexHull
                        points = np.column_stack((x_coords, y_coords))
                        
                        try:
                            # Compute convex hull
                            hull = ConvexHull(points)
                            
                            # Get hull vertices
                            hull_points = points[hull.vertices]
                            
                            # Create a new mask with the convex hull (uint8 for cv2)
                            fg_mask_temp = np.zeros((H, W), dtype=np.uint8)
                            
                            # Convert hull points to integer coordinates for cv2.fillPoly
                            hull_points_int = hull_points.astype(np.int32)
                            
                            # Fill the convex hull polygon
                            cv2.fillPoly(fg_mask_temp, [hull_points_int], 1)
                            
                            # Convert to boolean mask
                            fg_mask = fg_mask_temp.astype(bool)
                        except Exception as e:
                            # If convex hull computation fails, keep the original mask
                            # This catches any remaining edge cases
                            fg_mask = fg_mask_raw.copy()
                # else: keep the original mask if not enough points
        
        fg_masks.append(fg_mask)
    
    # Combine all foreground masks
    fg_mask = np.logical_or.reduce(fg_masks) if fg_masks else np.zeros_like(seg_frame, dtype=bool)
    bg_mask = np.logical_not(fg_mask)

    if no_background_condition:
        # Average the values of fg and bg for friction and bounciness
        for i in range(len(fg_static_friction_vals)):
            if fg_static_friction_vals[i] != 1.0:
                fg_static_friction_vals[i] = (fg_static_friction_vals[i] + bg_static_friction_val) / 2.0
            if fg_bounciness_vals[i] != 0.0:
                fg_bounciness_vals[i] = (fg_bounciness_vals[i] + bg_bounciness_vals[i]) / 2.0
        bg_static_friction_val = 1.0
        bg_bounciness_val = 0.0

    # Apply background values first
    if physprop_type == 'mass_only' or physprop_type == 'mass':
        physprop_frame[bg_mask, :] = bg_mass_val
    elif physprop_type == 'friction_only' or physprop_type == 'friction':
        physprop_frame[bg_mask, :] = bg_static_friction_val
    elif physprop_type == 'bounciness_only' or physprop_type == 'bounciness':
        physprop_frame[bg_mask, :] = bg_bounciness_val
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        physprop_frame[bg_mask, 0] = bg_bounciness_val
        physprop_frame[bg_mask, 1] = bg_static_friction_val
        physprop_frame[bg_mask, 2] = 1.0
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" in physprop_type:
        physprop_frame[bg_mask, 0] = bg_bounciness_val
        physprop_frame[bg_mask, 1] = bg_static_friction_val
        physprop_frame[bg_mask, 2] = 1.0
    else:
        raise ValueError(f"Invalid physprop type: {physprop_type}")
    
    # Apply foreground values for each object individually
    for i, (fg_mask_individual, fg_mass_val, fg_bounciness_val, fg_static_friction_val) in enumerate(zip(fg_masks, fg_mass_vals, fg_bounciness_vals, fg_static_friction_vals)):
        if physprop_type == 'mass_only' or physprop_type == 'mass':
            physprop_frame[fg_mask_individual, :] = fg_mass_val
        elif physprop_type == 'friction_only' or physprop_type == 'friction':
            physprop_frame[fg_mask_individual, :] = fg_static_friction_val
        elif physprop_type == 'bounciness_only' or physprop_type == 'bounciness':
            physprop_frame[fg_mask_individual, :] = fg_bounciness_val
        elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
            physprop_frame[fg_mask_individual, 0] = fg_bounciness_val
            physprop_frame[fg_mask_individual, 1] = fg_static_friction_val
            physprop_frame[fg_mask_individual, 2] = 1.0
        elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" in physprop_type:
            physprop_frame[fg_mask_individual, 0] = fg_bounciness_val
            physprop_frame[fg_mask_individual, 1] = fg_static_friction_val
            physprop_frame[fg_mask_individual, 2] = fg_mass_val
    
    if "deformable" in physprop_type:
        physprop_frame2[bg_mask, 0] = bg_neo_hookean_mu_val
        physprop_frame2[bg_mask, 1] = bg_neo_hookean_damping_val
        
        # Apply deformable properties for each foreground object individually
        for i, (fg_mask_individual, fg_neo_hookean_mu_val, fg_neo_hookean_damping_val) in enumerate(zip(fg_masks, fg_neo_hookean_mu_vals, fg_neo_hookean_damping_vals)):
            physprop_frame2[fg_mask_individual, 0] = fg_neo_hookean_mu_val
            physprop_frame2[fg_mask_individual, 1] = fg_neo_hookean_damping_val

    if "deformable" in physprop_type:
        physprop_frame = np.concatenate([physprop_frame, physprop_frame2], axis=-1)
    
    if "force" in physprop_type:
        # for i, (fg_mask_individual, fg_force_magnitude_val) in enumerate(zip(fg_masks, fg_force_magnitude_val)):
        move_obj_mask = fg_masks[fg_obj_ids.index(move_obj_id)]
        physprop_frame3[move_obj_mask, 0] = fg_force_magnitude_val

    if "force" in physprop_type or "move_dir" in physprop_type:   
        # for i, (fg_mask_individual, fg_dir_sin_theta_val, fg_dir_cos_theta_val) in enumerate(zip(fg_masks, move_dir_sin_theta_val, move_dir_cos_theta_val)):
        move_obj_mask = fg_masks[fg_obj_ids.index(move_obj_id)]
        physprop_frame3[move_obj_mask, 1] = move_dir_sin_theta_val
        physprop_frame3[move_obj_mask, 2] = move_dir_cos_theta_val
        
    if "force" in physprop_type or "move_dir" in physprop_type:
        physprop_frame = np.concatenate([physprop_frame, physprop_frame3], axis=-1)
        
    if return_physprop_text_labels:
        physprop_text_labels = {}
        mass_mean = np.mean(fg_mass_vals)
        physprop_text_labels["mass"] = "low" if mass_mean < 0.5 else "high"
        bounciness_mean = (np.mean(fg_bounciness_vals) + np.mean(bg_bounciness_val)) / 2.0
        physprop_text_labels["bounciness"] = "low" if bounciness_mean < 0.5 else "high"
        friction_mean = (np.mean(fg_static_friction_vals) + np.mean(bg_static_friction_val)) / 2.0
        physprop_text_labels["friction"] = "low" if friction_mean < 0.5 else "high"
        return physprop_frame, physprop_text_labels

    ## NOTE: when return_physprop_text_labels is True, return_neg_physprop should be False
    if return_neg_physprop:
        # Apply background values first
        if physprop_type == 'bounciness_only' or physprop_type == 'bounciness':
            neg_physprop_frame[bg_mask, 0] = neg_bg_bounciness_val
        elif physprop_type == 'friction_only' or physprop_type == 'friction':
            neg_physprop_frame[bg_mask, 0] = neg_bg_static_friction_val
        elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" in physprop_type:
            neg_physprop_frame[bg_mask, 0] = neg_bg_bounciness_val
            neg_physprop_frame[bg_mask, 1] = neg_bg_static_friction_val
            neg_physprop_frame[bg_mask, 2] = 1.0
        elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
            neg_physprop_frame[bg_mask, 0] = neg_bg_bounciness_val
            neg_physprop_frame[bg_mask, 1] = neg_bg_static_friction_val
            neg_physprop_frame[bg_mask, 2] = 1.0
        else:
            raise ValueError(f"Invalid physprop type: {physprop_type}")
        
        # Apply foreground values for each object individually
        for i, (fg_mask_individual, neg_fg_mass_val, neg_fg_bounciness_val, neg_fg_static_friction_val) in enumerate(zip(fg_masks, neg_fg_mass_vals, neg_fg_bounciness_vals, neg_fg_static_friction_vals)):
            if physprop_type == 'bounciness_only' or physprop_type == 'bounciness':
                neg_physprop_frame[fg_mask_individual, 0] = neg_fg_bounciness_val
            elif physprop_type == 'friction_only' or physprop_type == 'friction':
                neg_physprop_frame[fg_mask_individual, 0] = neg_fg_static_friction_val
            elif physprop_type == 'all':
                neg_physprop_frame[fg_mask_individual, 0] = neg_fg_mass_val
                neg_physprop_frame[fg_mask_individual, 1] = neg_fg_bounciness_val
                neg_physprop_frame[fg_mask_individual, 2] = neg_fg_static_friction_val
            elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
                neg_physprop_frame[fg_mask_individual, 0] = neg_fg_bounciness_val
                neg_physprop_frame[fg_mask_individual, 1] = neg_fg_static_friction_val
                neg_physprop_frame[fg_mask_individual, 2] = 1.0
        
        if "deformable" in physprop_type:
            neg_physprop_frame2[bg_mask, 0] = bg_neo_hookean_mu_val
            neg_physprop_frame2[bg_mask, 1] = bg_neo_hookean_damping_val
            
            # Apply deformable properties for each foreground object individually
            for i, (fg_mask_individual, neg_fg_neo_hookean_mu_val, neg_fg_neo_hookean_damping_val) in enumerate(zip(fg_masks, neg_fg_neo_hookean_mu_vals, neg_fg_neo_hookean_damping_vals)):
                neg_physprop_frame2[fg_mask_individual, 0] = neg_fg_neo_hookean_mu_val
                neg_physprop_frame2[fg_mask_individual, 1] = neg_fg_neo_hookean_damping_val

        if "deformable" in physprop_type:
            neg_physprop_frame = np.concatenate([neg_physprop_frame, neg_physprop_frame2], axis=-1)
        
        return physprop_frame, neg_physprop_frame
    
    return physprop_frame

def save_physprop_as_image(physprop_frame, save_path=None, physprop_type='all', cmap='viridis'):
    """
    Visualize the predicted physical property maps as subplots with colorbars.

    Args:
        physprop_frame (np.ndarray): Array of shape (H, W, C) or (B, H, W, C).
        save_path (str, optional): If provided, saves the figure to this path.
        property_names (list of str, optional): Names for each property channel.
        cmap (str): Colormap to use for visualization.
    """
    # Handle batch dimension if present
    if physprop_frame.ndim == 4:
        # Take the first sample in the batch
        physprop_frame = physprop_frame[0]
    assert physprop_frame.ndim == 3, "Expected shape (C, H, W)"

    H, W, C = physprop_frame.shape
    property_ids = list(range(C))
    # Default property names if not provided
    if physprop_type == 'mass_only' or physprop_type == 'mass':
        property_names = ["Mass"]
        C = 1
        property_ids = [0]
    elif physprop_type == 'bounciness_only' or physprop_type == 'bounciness' or physprop_type == 'restitution':
        property_names = ["Bounciness"]
        C = 1
        property_ids = [0]
    elif physprop_type == 'friction_only' or physprop_type == 'friction':
        property_names = ["Friction"]
        C = 1
        property_ids = [0]
    elif physprop_type == 'all':
        property_names = ["Mass", "Bounciness", "Friction"]
        C = 3
        property_ids = [0, 1, 2]
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        property_names = ["Bounciness", "Friction"]
        C = 2
        property_ids = [0, 1]
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" in physprop_type:
        property_names = ["Bounciness", "Friction", "Mass"]
        C = 3
        property_ids = [0, 1, 2]
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")

    if "deformable" in physprop_type:
        property_names += ["Neo-Hookean Mu", "Neo-Hookean Damping"]
        C += 2
        property_ids += [3, 4]
    if "force" in physprop_type or "move_dir" in physprop_type:
        property_names += ["Force Magnitude", "Sin Theta", "Cos Theta"]
        C += 3
        property_ids += [6, 7, 8]

    fig, axes = plt.subplots(1, C, figsize=(5*C, 5), squeeze=False)
    for i in range(C):
        ax = axes[0, i]
        im = ax.imshow(physprop_frame[:, :, property_ids[i]], cmap=cmap)
        ax.set_title(property_names[i])
        ax.axis('off')
        # Add colorbar, handle nan values
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Value', rotation=270, labelpad=15)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def save_physprop_as_text(physprop_vector, save_path=None):
    """Save physprop vector as text file"""
    # Convert the numpy array to a list
    physprop_vector = physprop_vector.tolist()
    # Convert the list to a string with each element on a new line
    physprop_vector_str = "\n".join(str(x) for x in physprop_vector)
    with open(save_path, 'w') as f:
        f.write(physprop_vector_str)

def save_physprop_as_image_blob(physprop_frame, save_path=None, physprop_type='all', cmap='viridis'):
    """
    Save physical property blob data as both text file and image visualization.
    
    Args:
        physprop_frame (np.ndarray): Array of shape (H, W, C*2) where C is number of properties.
                                   Each property has 2 channels: [background, foreground].
        save_path (str, optional): Base path for saving files. If provided, saves:
                                  - Text file: {save_path}_physprop.txt
                                  - Image: {save_path}_foreground.png
        physprop_type (str): Type of physical properties ('all', 'mass', 'bounciness', 'friction', etc.)
        cmap (str): Colormap to use for visualization.
    """
    # Handle batch dimension if present
    if physprop_frame.ndim == 4:
        # Take the first sample in the batch
        physprop_frame = physprop_frame[0]
    assert physprop_frame.ndim == 3, "Expected shape (H, W, C*2)"
    
    H, W, C_total = physprop_frame.shape
    num_properties = C_total // 2
    property_ids = list(range(num_properties))
    
    # Determine property names based on physprop_type
    if physprop_type == 'mass_only' or physprop_type == 'mass':
        property_names = ["Mass"]
        num_properties = 1
        property_ids = [0]
    elif physprop_type == 'bounciness_only' or physprop_type == 'bounciness' or physprop_type == 'restitution':
        property_names = ["Bounciness"]
        num_properties = 1
        property_ids = [0]
    elif physprop_type == 'friction_only' or physprop_type == 'friction':
        property_names = ["Friction"]
        num_properties = 1
        property_ids = [0]
    elif physprop_type == 'all':
        property_names = ["Mass", "Bounciness", "Friction"]
        num_properties = 3
        property_ids = [0, 1, 2]
    elif "friction" in physprop_type and "bounciness" in physprop_type and "mass" not in physprop_type:
        property_names = ["Bounciness", "Friction"]
        num_properties = 2
        property_ids = [0, 1]
    else:
        raise ValueError(f"Invalid physprop_type: {physprop_type}")
    
    if "deformable" in physprop_type:
        property_names += ["Neo-Hookean Mu", "Neo-Hookean Damping"]
        num_properties += 2
        property_ids += [3, 4]
    
    # Extract foreground channels (every 2nd channel starting from index 1)
    foreground_channels = physprop_frame[:, :, 1::2]
    
    # Create physical property vector for text file
    # Format: [fg_mass, bg_mass, fg_bounciness, bg_bounciness, fg_friction, bg_friction]
    physprop_vector = []
    for i in range(num_properties):
        # Get foreground and background values (average over the object regions)
        fg_channel = physprop_frame[:, :, i*2 + 1]  # Foreground channel
        bg_channel = physprop_frame[:, :, i*2]      # Background channel
        
        # Calculate average values (excluding zeros for foreground if no object)
        fg_mask = fg_channel > 0
        if np.any(fg_mask):
            fg_val = np.mean(fg_channel[fg_mask])
        else:
            # Use default values if no foreground object
            if property_names[i] == "Mass":
                fg_val = 1.0
            elif property_names[i] == "Bounciness":
                fg_val = 0.0
            elif property_names[i] == "Friction":
                fg_val = 1.0
        
        bg_val = np.mean(bg_channel)
        
        physprop_vector.extend([fg_val, bg_val])
    
    # Save as text file
    if save_path is not None:
        txt_path = f"{save_path}_physprop.txt"
        save_physprop_as_text(np.array(physprop_vector), txt_path)
        print(f"Physical properties saved to: {txt_path}")
    
    # Create visualization of foreground channels
    fig, axes = plt.subplots(1, num_properties, figsize=(5*num_properties, 5), squeeze=False)
    
    for i in range(num_properties):
        ax = axes[0, i]
        im = ax.imshow(foreground_channels[:, :, i], cmap=cmap)
        ax.set_title(f"{property_names[i]} (Foreground)")
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Value', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save image if path provided
    if save_path is not None:
        img_path = f"{save_path}_foreground.png"
        plt.savefig(img_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Foreground visualization saved to: {img_path}")
    else:
        plt.show()
    
    return physprop_vector

def augment_input_image_with_move_dir(input_image, dir_angle, seg_frame, fg_seg_id, force_magnitude = None,save_path=None):
    """
    Augment the input image with the force direction.
    """
    import cv2
    import numpy as np

    H, W = input_image.shape[:2]
    center_y = H // 2
    center_x = W // 2
    fg_mask_raw = seg_frame == fg_seg_id
    fg_mask = fg_mask_raw.copy()
    if np.any(fg_mask):
        # Calculate object center
        y_coords, x_coords = np.where(fg_mask)
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))

    # Calculate the endpoint of the force vector
    length = 100
    end_x = int(round(center_x + length * np.cos(dir_angle)))
    end_y = int(round(center_y + length * np.sin(dir_angle)))

    # Convert the image to BGR
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # Draw a line along the force angle from the center of the object
    cv2.line(input_image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
    if force_magnitude is not None:
        cv2.putText(
            input_image,
            f"{force_magnitude}",
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    if save_path is not None:
        cv2.imwrite(save_path, input_image)
    return input_image


def get_physprop_as_text_labels(physprop_frame, physprop_type='all'):
    """Generate text labels for the physical properties"""

    # Handle batch dimension if present
    if physprop_frame.ndim == 4:
        # Take the first sample in the batch
        physprop_frame = physprop_frame[0]
    assert physprop_frame.ndim == 3, "Expected shape (C, H, W)"

    H, W, C = physprop_frame.shape
    property_ids = list(range(C))
    # Default property names if not provided

    property_ids = []
    prev_start_id = 0

    physprop_label_dict = {}
    if "bounciness" in physprop_type:
        property_ids += [prev_start_id]
        property_id = prev_start_id
        bounciness_mean = np.mean(np.unique(physprop_frame[:, :, property_id]))
        physprop_label_dict["bounciness"] = "low" if bounciness_mean < 0.5 else "high"
        prev_start_id += 1

    if "friction" in physprop_type:
        property_ids += [prev_start_id]
        property_id = prev_start_id
        friction_mean = np.mean(np.unique(physprop_frame[:, :, property_id]))
        physprop_label_dict["friction"] = "low" if friction_mean < 0.5 else "high"
        prev_start_id += 1
    
    if "mass" in physprop_type:
        property_ids += [prev_start_id]
        property_id = prev_start_id
        mass_mean = np.mean(np.unique(physprop_frame[:, :, property_id]))
        physprop_label_dict["mass"] = "low" if mass_mean < 0.5 else "high"
        prev_start_id += 1

    # if "deformable" in physprop_type:
    #     property_names += ["neo-hookean mu", "neo-hookean damping"]
    #     property_ids += [prev_start_id, prev_start_id + 1]
    #     prev_start_id += 3
    # if "force" in physprop_type or "move_dir" in physprop_type:
    #     property_names += ["force magnitude", "sin theta", "cos theta"]
    #     property_ids += [prev_start_id, prev_start_id + 1, prev_start_id + 2]
    #     prev_start_id += 3
    # elif "move_dir" in physprop_type:
    #     property_names += ["sin theta", "cos theta"]
    #     property_ids += [prev_start_id + 1, prev_start_id + 2]
    #     prev_start_id += 3

    return physprop_label_dict

def get_object_movement_direction_text(start_coords, end_coords):
    # The direction text could be one of left, right, up, down, up-left, up-right, down-left, down-right
    # Calculate direction based on pixel coordinates
    dx = end_coords[0] - start_coords[0]
    dy = end_coords[1] - start_coords[1]
    
    # Use a threshold to determine if movement is mostly horizontal, vertical, or diagonal
    threshold = 0.3  # 30% threshold for diagonal classification
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    
    if abs(dx) < 1e-6:  # Pure vertical movement
        direction_text = "down" if dy > 0 else "up"
    elif abs(dy) < 1e-6:  # Pure horizontal movement
        direction_text = "right" if dx > 0 else "left"
    else:
        # Check if diagonal or primarily one direction
        ratio = abs_dx / (abs_dx + abs_dy)
        
        if ratio < threshold:  # Primarily vertical
            direction_text = "down" if dy > 0 else "up"
        elif ratio > (1 - threshold):  # Primarily horizontal
            direction_text = "right" if dx > 0 else "left"
        else:  # Diagonal movement
            if dx > 0 and dy > 0:
                direction_text = "down-right"
            elif dx > 0 and dy < 0:
                direction_text = "up-right"
            elif dx < 0 and dy > 0:
                direction_text = "down-left"
            else:
                direction_text = "up-left"
    return direction_text


def parse_metadata_for_props(metadata: Dict[str, Any], fg_seg_ids: Optional[List[int]], bg_seg_id: Optional[int]) -> Dict[str, Any]:

    seg_ids = metadata['segmentation_id']
    # Seg ids can be something like [5, 7, 1, 2]
    # We need to map them to [2, 3, 0, 1]
    object_ids = np.zeros_like(seg_ids)
    for i, seg_id in enumerate(seg_ids):
        object_ids[i] = seg_ids.index(seg_id)

    force_magnitude = 0.0
    if "applied_forces_image" in metadata:
        force_information = metadata["applied_forces_image"][0]
        force_magnitude = force_information["force_magnitude"]
        force_magnitude_min = metadata["min_force"]
        force_magnitude_max = metadata["max_force"]
        force_magnitude = (force_magnitude - force_magnitude_min) / (force_magnitude_max - force_magnitude_min)

    mass = metadata['mass']
    bounciness = metadata['restitution']
    static_friction = metadata['friction']
    neo_hookean_mu = metadata.get('neo_hookean_mu', None)
    neo_hookean_lambda = metadata.get('neo_hookean_lambda', None)
    neo_hookean_damping = metadata.get('neo_hookean_damping', None)
    neo_hookean_mu_min = metadata.get('neo_hookean_mu_min', 60)
    neo_hookean_mu_max = metadata.get('neo_hookean_mu_max', 600)
    neo_hookean_lambda_min = metadata.get('neo_hookean_lambda_min', 100)
    neo_hookean_lambda_max = metadata.get('neo_hookean_lambda_max', 600)
    raw_neo_hookean_mu_default_val = 600.0
    raw_neo_hookean_lambda_default_val = 600.0


    if fg_seg_ids is not None:
        # Multiple foreground objects specified
        fg_obj_ids = [seg_ids.index(seg_id) for seg_id in fg_seg_ids if seg_id in seg_ids]
        if len(fg_obj_ids) == 0:
            raise ValueError("None of the specified fg_seg_ids found in segmentation_id")
    else:
        # Default behavior - use max object ID
        fg_obj_ids = [np.max(object_ids)]
    
    move_obj_id = fg_obj_ids[0]

    # Determine background object ID
    if bg_seg_id is not None:
        bg_obj_id = seg_ids.index(bg_seg_id)
    else:
        # Use the first foreground object ID - 1 as default background
        bg_obj_id = fg_obj_ids[0] - 1

    fg_static_friction_vals = [np.clip(static_friction[obj_id], 0.0, 1.0) for obj_id in fg_obj_ids]
    fg_bounciness_vals = [np.clip(bounciness[obj_id], 0.0, 1.0) for obj_id in fg_obj_ids]
    fg_mass_vals = [1.0 for obj_id in fg_obj_ids]

    bg_static_friction_val = np.clip(static_friction[bg_obj_id], 0.0, 1.0)
    bg_bounciness_val = np.clip(bounciness[bg_obj_id], 0.0, 1.0)
    bg_mass_val = 1.0

    props = {
        "fg_static_friction_vals": fg_static_friction_vals,
        "fg_bounciness_vals": fg_bounciness_vals,
        "fg_mass_vals": fg_mass_vals,
        "bg_static_friction_val": bg_static_friction_val,
        "bg_bounciness_val": bg_bounciness_val,
        "bg_mass_val": bg_mass_val,
        "force_magnitude": force_magnitude,
    }

    if neo_hookean_mu is not None:
        neo_hookean_mu = [mu_val if mu_val is not None else raw_neo_hookean_mu_default_val for mu_val in neo_hookean_mu]
        neo_hookean_mu = (np.array(neo_hookean_mu) - neo_hookean_mu_min) / (neo_hookean_mu_max - neo_hookean_mu_min)
    else:
        neo_hookean_mu = [1.0 for _ in range(len(mass))]
    if neo_hookean_lambda is not None:
        neo_hookean_lambda = [lambda_val if lambda_val is not None else raw_neo_hookean_lambda_default_val for lambda_val in neo_hookean_lambda]
        neo_hookean_lambda = (np.array(neo_hookean_lambda) - neo_hookean_lambda_min) / (neo_hookean_lambda_max - neo_hookean_lambda_min)
    else:
        neo_hookean_lambda = [1.0 for _ in range(len(mass))]
    if neo_hookean_damping is not None:
        neo_hookean_damping = [damping_val if damping_val is not None else 1.0 for damping_val in neo_hookean_damping]
    else:
        neo_hookean_damping = [1.0 for _ in range(len(mass))]
    neo_hookean_damping = np.array(neo_hookean_damping)
    fg_neo_hookean_mu_vals = [neo_hookean_mu[obj_id] if not np.isnan(neo_hookean_mu[obj_id]) else 1.0 for obj_id in fg_obj_ids]
    bg_neo_hookean_mu_val = neo_hookean_mu[bg_obj_id] if not np.isnan(neo_hookean_mu[bg_obj_id]) else 1.0
    fg_neo_hookean_lambda_vals = [neo_hookean_lambda[obj_id] if not np.isnan(neo_hookean_lambda[obj_id]) else 1.0 for obj_id in fg_obj_ids]
    bg_neo_hookean_lambda_val = neo_hookean_lambda[bg_obj_id] if not np.isnan(neo_hookean_lambda[bg_obj_id]) else 1.0
    fg_neo_hookean_damping_vals = [neo_hookean_damping[obj_id] if not np.isnan(neo_hookean_damping[obj_id]) else 1.0 for obj_id in fg_obj_ids]
    bg_neo_hookean_damping_val = neo_hookean_damping[bg_obj_id] if not np.isnan(neo_hookean_damping[bg_obj_id]) else 1.0

    props["fg_neo_hookean_mu_vals"] = fg_neo_hookean_mu_vals
    props["bg_neo_hookean_mu_val"] = bg_neo_hookean_mu_val
    props["fg_neo_hookean_lambda_vals"] = fg_neo_hookean_lambda_vals
    props["bg_neo_hookean_lambda_val"] = bg_neo_hookean_lambda_val
    props["fg_neo_hookean_damping_vals"] = fg_neo_hookean_damping_vals
    props["bg_neo_hookean_damping_val"] = bg_neo_hookean_damping_val
    return props

def parse_props_for_questionnaire(props):
    
    fg_friction_mean = np.mean(props["fg_static_friction_vals"])
    bg_friction_mean = props["bg_static_friction_val"]
    avg_friction_mean = (fg_friction_mean + bg_friction_mean) / 2.0
    fg_bounciness_mean = np.mean(props["fg_bounciness_vals"])
    bg_bounciness_mean = props["bg_bounciness_val"]
    avg_bounciness_mean = (fg_bounciness_mean + bg_bounciness_mean) / 2.0
    fg_neo_hookean_mu_vals = np.mean(props["fg_neo_hookean_mu_vals"])
    fg_neo_hookean_lambda_vals = props["fg_neo_hookean_lambda_vals"]
    fg_neo_hookean_damping_vals = np.mean(props["fg_neo_hookean_damping_vals"])
    deformability = (fg_neo_hookean_mu_vals + fg_neo_hookean_damping_vals) / 2.0
    return {
        "friction": avg_friction_mean,
        "bounciness": avg_bounciness_mean,
        "deformability": deformability,
        "force_magnitude": props["force_magnitude"],
    }