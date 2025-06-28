import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from database import MongoDBClient, MinioClient
from config import Config

class ImageProcessor:
    def __init__(self):
        self.mongo = MongoDBClient()
        self.minio = MinioClient()

    def custom_colormap(self, image):
        colored = plt.get_cmap('plasma')(image / 255.0)[:, :, :3]
        return (colored * 255).astype(np.uint8)

    def process_and_upload(self, csv_path):
        df = pd.read_csv(csv_path)
        df.interpolate(axis=0, inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        depths = df['depth'].values
        image_data = df.drop(columns=['depth']).values

        resized_rows = []
        for row in image_data:
            resized_row = cv2.resize(row.reshape(1, -1), (150, 1), interpolation=cv2.INTER_AREA)
            resized_rows.append(resized_row.flatten())

        resized_image = np.vstack(resized_rows).astype(np.uint8)
        colored_image = self.custom_colormap(resized_image)

        for idx, depth in enumerate(depths):
            single_row = colored_image[idx:idx+1, :, :]
            _, buffer = cv2.imencode('.png', single_row)
            image_bytes = io.BytesIO(buffer.tobytes())

            # Upload to MinIO
            object_name = f"frame_{depth}.png"
            self.minio.client.put_object(
                Config.MINIO_BUCKET,
                object_name,
                image_bytes,
                length=image_bytes.getbuffer().nbytes,
                content_type='image/png'
            )

            # Store metadata in MongoDB
            self.mongo.collection.insert_one({
                "depth": float(depth),
                "minio_object": object_name
            })


if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_and_upload("Challenge2.csv")
