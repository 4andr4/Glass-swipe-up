let device, context, bgPipeline, glassPipeline, uniformBuffer;

let dragProgress = 0.0;       
let targetDragProgress = 0.0; 

let swayProgress = 0.0;       
let targetSwayProgress = 0.0; 

let isDragging = false;
let isOpen = false;           
let lastDragY = 0.0;          
let dragVelocity = 0.0;

let aspectRatio = 1.0;

let dragOffset = 0.0;
let swayOffset = 0.0;
let sensitivity = 0.7;

// vertex shader 📌
const commonVertexShader = `
        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex
        fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {
            var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            var out: VertexOutput;
            out.pos = vec4f(pos[idx], 0.0, 1.0);
            out.uv = pos[idx] * 0.5 + 0.5;
            return out;
        }
    `;


// background shader 🧙
const bgShader = commonVertexShader + `
    struct Uniforms { time: f32, dragProgress: f32, swayProgress: f32, aspectRatio: f32 };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    @fragment
    fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
        let worldP = (uv * 2.0 - 1.0) * vec2f(u.aspectRatio, 1.0);
        let lineY = 0.8;
        let lineThickness = 0.03; 
        let dist = abs(worldP.y - lineY);
        
        // Maska: 1.0 na črti, 0.0 drugje
        let mask = 1.0 - smoothstep(lineThickness - 0.002, lineThickness, dist);
        
        let lineColor = mix(vec3f(0, 0.529, 1), vec3f(0.1, 0.529, 1), uv.y);
        
        return vec4f(lineColor, mask);
    }
`;

// glass shader 🧙
const glassShader = `
struct Uniforms { time: f32, dragProgress: f32, swayProgress: f32, aspectRatio: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var bgTex: texture_2d<f32>;
@group(0) @binding(2) var bgSamp: sampler;

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let glassColor = vec3f(0.929, 0.937, 0.949);
    let customColorBottom = vec3f(0.008, 0.843, 1); 
    let blurBase = 0.015; 
    let distortion = 0.28; 
    let samples = 40;

    let p = (uv * 2.0 - 1.0) * vec2f(u.aspectRatio, 1.0);
    let center = vec2f(u.swayProgress * 0.1, -1.3 + u.dragProgress * 2.2);
    
    let shrink = 1.0 - smoothstep(0.0, 0.7, u.dragProgress) * 0.8;
    let expand = smoothstep(0.9, 1.0, u.dragProgress);
    
    let targetWidth = 0.7; 
    let targetHeight = 0.06;
    let sizeX = mix(0.78 * shrink, targetWidth, expand);
    let sizeY = mix(0.78 * shrink, targetHeight, expand);

    let roundness = mix(sizeY, 0.057, expand); 

    let q = abs(p - center) - vec2f(sizeX, sizeY) + roundness;
    let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - roundness;

    var finalColor = vec3f(0.0);
    var alpha = 0.0;

    if (d <= 0.0) {
        // Popravek za localP, da so efekti pravilno poravnani
        let localP = (p - center) / vec2f(sizeX, sizeY);
        let distNorm = length(localP);
        
        let currentDistortion = distortion * (1.0 - expand * 0.7);
        let bentUV = uv - localP * pow(distNorm, 1.6) * currentDistortion;
        
        var finalAcc = vec3f(0.0);
        let rnd = hash(fragCoord.xy);

        for (var i = 0; i < samples; i++) {
            let t = f32(i) / f32(samples);
            let angle = t * 35.0 + rnd * 6.28;
            let r = sqrt(t) * blurBase * distNorm;
            let offset = vec2f(cos(angle), sin(angle)) * r;
            let rawSample = textureSampleLevel(bgTex, bgSamp, bentUV + offset, 0.0).rgb;
            
            let brightness = max(rawSample.r, max(rawSample.g, rawSample.b));
            let tintedBottom = customColorBottom * brightness * 1.5;
            let isLowerSample = smoothstep(0.0, 0.5, -offset.y * 80.0);
            let sampleColor = mix(rawSample, tintedBottom, isLowerSample * (1.0 - expand));
            finalAcc += sampleColor;
        }

        let blurredBG = finalAcc / f32(samples);
        var menuColor = mix(blurredBG, glassColor, 0.1);
        menuColor += glassColor * (1.0 - max(blurredBG.r, max(blurredBG.g, blurredBG.b))) * 0.89;

        // Rob (Rim light) - d/sizeY poskrbi, da je rob enakomeren
        let rim = smoothstep(-0.05, 0.0, d);
        menuColor = mix(menuColor, vec3f(1), rim * (0.7 - expand * 0.0));

        let edgeSoftness = 0.002;
        let aa = smoothstep(edgeSoftness, -edgeSoftness, d);
        
        finalColor = menuColor;
        alpha = aa; 
    }

    return vec4f(finalColor, alpha);
}
`;

async function initWebGPU(canvasId) {

    // priprava 📝
    const canvas = document.getElementById(canvasId);
    if (!navigator.gpu) { console.error("WebGPU ni podprt."); return; }

    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({ device, format, alphaMode: 'premultiplied' });


    // buffer 📦
    uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });


    // tekstura ozadja 🖼️
    let bgTexture, bgView;
    function createBGTexture() {
        bgTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: navigator.gpu.getPreferredCanvasFormat(),
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        bgView = bgTexture.createView();
    }

    // sampler
    const bgSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
    });



    // zapakiramo shader-je 📦✅
    const bgModule = device.createShaderModule({ code: bgShader});
    const glassModule = device.createShaderModule({ code: glassShader });



    const transparentBlend = {
            color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            },
            alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            }
        };


    // bgPipeline 🖼️👷
    bgPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { 
                module: bgModule, 
                entryPoint: 'main' 
            },
            fragment: { 
                module: bgModule, 
                entryPoint: 'fs_main', 
                targets: [{ 
                    format: format,
                    blend: transparentBlend
                }] 
            },
            primitive: { topology: 'triangle-list' }
        });

    // glassPipeline 🔍👷
    glassPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { 
                module: device.createShaderModule({ code: commonVertexShader }), 
                entryPoint: 'main' 
            },
            fragment: { 
                module: glassModule, 
                entryPoint: 'fs_main', 
                targets: [{ 
                    format: format,
                    blend: transparentBlend
                }] 
            },
            primitive: { topology: 'triangle-list' }
        });




    // bind group-i
    function updateBindGroups() {

        if (!bgPipeline || !glassPipeline) return;

        bgBindGroup = device.createBindGroup({
            layout: bgPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
        });

        glassBindGroup = device.createBindGroup({
            layout: glassPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: bgView },
                { binding: 2, resource: bgSampler }
            ]
        });
    }


    function resize() {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = window.innerWidth * dpr;
        canvas.height = window.innerHeight * dpr;
        aspectRatio = canvas.width / canvas.height;
        createBGTexture();
        updateBindGroups();
    }
    window.addEventListener('resize', resize);

    resize();




    // Zaznavanje klika 🖱️
    function isPointInBlob(clientX, clientY) {
        let ndcX = ((clientX / window.innerWidth) * 2.0 - 1.0) * aspectRatio;
        let ndcY = -((clientY / window.innerHeight) * 2.0 - 1.0);

        let animY = -1.3 + dragProgress * 2.2;
        let animX = swayProgress * 0.3;

        let currentWidth = 0.45 * (1.0 - dragProgress) + 0.8 * dragProgress; 
        let currentHeight = 0.45 * (1.0 - dragProgress) + 0.3 * dragProgress; 

        let dx = Math.abs(ndcX - animX);
        let dy = Math.abs(ndcY - animY);

        return dx < (currentWidth + 0.1) && dy < (currentHeight + 0.15);
    }

    // handle input
    function handleInput(clientX, clientY) {

        let rawMouseY = 1.0 - (clientY / window.innerHeight);
        let rawMouseX = (clientX / window.innerWidth) * 2.0 - 1.0;

        let desiredDrag = (rawMouseY + dragOffset) * sensitivity;
        let desiredSway = rawMouseX + swayOffset;

        targetDragProgress = Math.max(0.0, Math.min(1.0, desiredDrag)); 
        targetSwayProgress = desiredSway * 1.2;
    }

    function handleDragEnd() {
        if (!isDragging) return;
        isDragging = false;
        
        // snap 🧲
        const snapThreshold = 0.5; 
        
        if (dragVelocity > 0.01) {
            targetDragProgress = 1.0;
            isOpen = true;
        } else if (dragVelocity < -0.01) {
            targetDragProgress = 0.0;
            isOpen = false;
        } else {
            if (dragProgress > snapThreshold) {
                targetDragProgress = 1.0;
                isOpen = true;
            } else {
                targetDragProgress = 0.0;
                isOpen = false;
            }
        }
        
        targetSwayProgress = 0.0;
        console.log(isOpen ? "Stanje: ODPRTO" : "Stanje: ZAPRTO");
    }


    window.addEventListener('mousedown', (e) => { 
        if (!isPointInBlob(e.clientX, e.clientY)) return;

        isDragging = true; 
        lastDragY = dragProgress;

        let rawMouseY = 1.0 - (e.clientY / window.innerHeight);
        let rawMouseX = (e.clientX / window.innerWidth) * 2.0 - 1.0;

        dragOffset = (dragProgress / sensitivity) - rawMouseY;
        
        swayOffset = (swayProgress / 1.2) - rawMouseX;
    });

    window.addEventListener('mousemove', (e) => {
        if (isDragging) {
            handleInput(e.clientX, e.clientY);
            dragVelocity = dragProgress - lastDragY;
            lastDragY = dragProgress;
        }
    });

    window.addEventListener('mouseup', handleDragEnd);

    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        
        isDragging = false;
        const threshold = 0.3;
        
        if (dragProgress > threshold || dragVelocity > 0.05) {
            targetDragProgress = 1.0;
            isOpen = true;
            console.log("Stanje: ODPRTO");
        } else {
            targetDragProgress = 0.0;
            isOpen = false;
        }
        targetSwayProgress = 0.0;
    });

    window.addEventListener('touchstart', (e) => {
        let touch = e.touches[0];
        if (isPointInBlob(touch.clientX, touch.clientY)) {
            isDragging = true;
            lastDragY = dragProgress;

            let rawMouseY = 1.0 - (touch.clientY / window.innerHeight);
            let rawMouseX = (touch.clientX / window.innerWidth) * 2.0 - 1.0;
            
            dragOffset = (dragProgress / 0.85) - rawMouseY;
            swayOffset = (swayProgress / 1.2) - rawMouseX;
        }
    });

    window.addEventListener('touchend', () => {
        isDragging = false;
        if (dragProgress > 0.4 || dragVelocity > 0.02) {
            targetDragProgress = 1.0;
            isOpen = true;
        } else {
            targetDragProgress = 0.0;
            isOpen = false;
        }
        targetSwayProgress = 0.0;
    });

    window.addEventListener('touchmove', (e) => {
        if(isDragging) handleInput(e.touches[0].clientX, e.touches[0].clientY);
    });






    // RENDER LOOP ♾️
    function render(time) {

        let speed = isDragging ? 0.09 : 0.02;

        dragProgress += (targetDragProgress - dragProgress) * speed;
        swayProgress += (targetSwayProgress - swayProgress) * speed;

        document.body.style.setProperty('--blob-p', dragProgress);

        const uniformData = new Float32Array([
            time * 0.001, 
            dragProgress, 
            swayProgress, 
            aspectRatio
        ]);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();


        // PASS 1
        const pass1 = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: bgView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                    loadOp: 'clear', 
                    storeOp: 'store'
                }]
            });
        pass1.setPipeline(bgPipeline);
        pass1.setBindGroup(0, bgBindGroup);
        pass1.draw(3);
        pass1.end();


        // PASS 2
        const pass2 = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                    loadOp: 'clear', storeOp: 'store'
                }]
            });
        pass2.setPipeline(glassPipeline);
        pass2.setBindGroup(0, glassBindGroup);
        pass2.draw(3);
        pass2.end();



        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);





        // Skrij html text, ko sphere zaključi 🙈
        const mainContent = document.querySelector('.home-content');
        if (mainContent) {
            mainContent.style.pointerEvents = dragProgress > 0.9 ? 'none' : 'auto';
        }

    }
    requestAnimationFrame(render);
}

window.startMyWebGPU = initWebGPU;  

