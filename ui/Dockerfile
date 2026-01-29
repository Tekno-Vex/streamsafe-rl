# Minimal Node Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package.json .
# We use 'true' so it passes even without real dependencies yet
RUN echo "Skipping install for placeholder" 
CMD ["npm", "start"]