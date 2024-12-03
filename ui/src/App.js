import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  ChakraProvider,
  Box,
  VStack,
  Heading,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Button,
  useToast,
  Text,
  List,
  ListItem,
  Progress,
  Tag,
} from '@chakra-ui/react';

const API_URL = 'http://localhost:8000/api/v1';

function App() {
  const [manuscript, setManuscript] = useState({
    title: '',
    abstract: '',
    content: '',
    authors: '',
    keywords: '',
    discipline: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [reviewResult, setReviewResult] = useState(null);
  const toast = useToast();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setManuscript((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const response = await axios.post(`${API_URL}/submit`, {
        ...manuscript,
        authors: manuscript.authors.split(',').map((a) => a.trim()),
        keywords: manuscript.keywords.split(',').map((k) => k.trim()),
      });

      setReviewResult(response.data);
      toast({
        title: 'Manuscript submitted successfully',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error submitting manuscript',
        description: error.response?.data?.detail || 'Something went wrong',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <ChakraProvider>
      <Box p={8} maxWidth="800px" mx="auto">
        <VStack spacing={8} align="stretch">
          <Heading>AI Peer Review System</Heading>

          <form onSubmit={handleSubmit}>
            <VStack spacing={4} align="stretch">
              <FormControl isRequired>
                <FormLabel>Title</FormLabel>
                <Input
                  name="title"
                  value={manuscript.title}
                  onChange={handleInputChange}
                  placeholder="Enter manuscript title"
                />
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Abstract</FormLabel>
                <Textarea
                  name="abstract"
                  value={manuscript.abstract}
                  onChange={handleInputChange}
                  placeholder="Enter manuscript abstract"
                  rows={4}
                />
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Content</FormLabel>
                <Textarea
                  name="content"
                  value={manuscript.content}
                  onChange={handleInputChange}
                  placeholder="Enter manuscript content"
                  rows={10}
                />
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Authors (comma-separated)</FormLabel>
                <Input
                  name="authors"
                  value={manuscript.authors}
                  onChange={handleInputChange}
                  placeholder="John Doe, Jane Smith"
                />
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Keywords (comma-separated)</FormLabel>
                <Input
                  name="keywords"
                  value={manuscript.keywords}
                  onChange={handleInputChange}
                  placeholder="AI, Machine Learning, Research"
                />
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Discipline</FormLabel>
                <Input
                  name="discipline"
                  value={manuscript.discipline}
                  onChange={handleInputChange}
                  placeholder="Computer Science"
                />
              </FormControl>

              <Button
                type="submit"
                colorScheme="blue"
                isLoading={isSubmitting}
                loadingText="Submitting"
              >
                Submit for Review
              </Button>
            </VStack>
          </form>

          {reviewResult && (
            <Box mt={8} p={4} borderWidth={1} borderRadius="md">
              <Heading size="md" mb={4}>
                Review Results
              </Heading>
              
              <Text mb={2}>
                Manuscript ID: <Tag>{reviewResult.manuscript_id}</Tag>
              </Text>
              
              <Text mb={2}>Overall Score:</Text>
              <Progress
                value={reviewResult.overall_score * 100}
                colorScheme={
                  reviewResult.overall_score > 0.7
                    ? 'green'
                    : reviewResult.overall_score > 0.4
                    ? 'yellow'
                    : 'red'
                }
                mb={4}
              />

              <Text mb={2}>Suggestions:</Text>
              <List spacing={2} mb={4}>
                {reviewResult.suggestions.map((suggestion, index) => (
                  <ListItem key={index}>• {suggestion}</ListItem>
                ))}
              </List>

              <Text mb={2}>Agent Reviews:</Text>
              {Object.entries(reviewResult.agent_reviews).map(([agent, review]) => (
                <Box key={agent} p={2} bg="gray.50" borderRadius="md" mb={2}>
                  <Text fontWeight="bold">{agent}</Text>
                  <Text>Score: {review.score.toFixed(2)}</Text>
                  <Text>Confidence: {review.confidence.toFixed(2)}</Text>
                  <List>
                    {review.comments.map((comment, index) => (
                      <ListItem key={index}>• {comment}</ListItem>
                    ))}
                  </List>
                </Box>
              ))}
            </Box>
          )}
        </VStack>
      </Box>
    </ChakraProvider>
  );
}

export default App; 