"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""


def test_offset_correctly_augmented_2D():
    """Check whether the offset images are correctly augmented.
    The offset images are N dimensional, where at each cell center at t an
    offset vector to the corresponding cell center at the previous
     frame t-1 is stored"""
    import numpy as np

    np.random.seed(42)
    from embedtrack.utils.utils import get_indices_pandas
    from embedtrack.utils.transforms import RandomRotationsAndFlips

    augmentation = RandomRotationsAndFlips(degrees=90)
    for i in range(100):
        dummy_shape = (300, 300)
        dummy_1 = np.zeros(dummy_shape)
        dummy_2 = np.zeros(dummy_shape)
        inst_pos_1 = np.random.choice(dummy_1.size, 50, False)
        inst_pos_2 = np.random.choice(dummy_2.size, 50, False)

        obj_ids_1 = {}
        obj_ids_2 = {}
        for label, (pos_1, pos_2) in enumerate(zip(inst_pos_1, inst_pos_2)):
            index_1 = np.unravel_index(pos_1, dummy_shape)
            index_2 = np.unravel_index(pos_2, dummy_shape)
            dummy_1[index_1] = label
            dummy_2[index_2] = label
            obj_ids_1[label] = index_1
            obj_ids_2[label] = index_2

        offset_img = np.zeros((*dummy_shape, len(dummy_shape)))  # hxwx2 or dxhxwx3
        for obj_id, seed_position in obj_ids_1.items():
            prev_position = obj_ids_2[obj_id]
            difference = np.array(seed_position) - np.array(prev_position)
            # shift all pixels by avg offset
            offset_img[seed_position[0], seed_position[1], :] = difference.reshape(
                1, -1
            )

        offset_img = np.transpose(offset_img, (2, 0, 1)).copy()
        data = {"seeds_1": dummy_1, "seeds_2": dummy_2, "flow": offset_img}
        augmented_data = augmentation(data)

        dummy_augmented = np.zeros(dummy_shape)
        prev_frame = augmented_data["seeds_2"]
        obj_ids_1_augmented = get_indices_pandas(augmented_data["seeds_1"])
        for label, position in obj_ids_1_augmented.items():
            prev_pos = (
                np.array(position)
                - augmented_data["flow"]
                .squeeze()[:, position[0], position[1]]
                .reshape(-1, 1)
            ).astype(np.int)
            dummy_augmented[tuple(prev_pos.reshape(-1))] = label
        assert (dummy_augmented - prev_frame).sum() == 0, "Wrong flipping"


if __name__ == "__main__":
    test_offset_correctly_augmented_2D()
