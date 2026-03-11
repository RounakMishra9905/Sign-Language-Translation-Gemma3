    def __getitem__(self, index):
        # Load the image and pose data
        image_path = self.image_paths[index]
        pose_data = self.pose_data[index]

        # Load the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Tokenize the input sentence
        tokenized_input = self.tokenizer.encode(self.sentences[index], return_tensors='pt')

        # Create input mask
        input_mask = (tokenized_input != self.tokenizer.pad_token_id).long()

        # Create labels with masking
        labels = tokenized_input.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens

        return {'image': image, 'pose_data': pose_data, 'input_ids': tokenized_input[0], 'attention_mask': input_mask[0], 'labels': labels[0]}