# Supporting Code for blog

Following the first code snippet in the blog post clones this repository and builds the Docker images from the Dockerfiles. While the Dockerfiles are meant to use the latest versions of included software, it is possible that newer changes to software will break. We have also included some additional lines (commented out with 4 hashes) which fix the third party software to the versions we used at the time of writing the blog. A simple way to get the docker files fixed to these is to run 
```bash
bash revert2fixed.sh pytorch.dockerfile
```
to create `pytorch.dockerfile-fixed` which could be used in the code in place of `pytorch.dockerfile`.
 However, this third-party software may also use un-pinned versions of software and that could still break reproducibility.
The code snippets posted on the blog are contained in the `jax_script.sh` and `torch_script.sh`. 
Finally, we remind the user that the code here is licensed through the license in this github repository.
