import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

real_label = 1
fake_label = 0
latent_dim = 256

visualized = False


def visualize(img, out):
    img = ToPILImage()(img[0])
    out = ToPILImage()(out[0])

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(img)
    ax[1].imshow(out)
    plt.tight_layout()
    ax[0].axis('off')
    ax[0].set_title('original')
    ax[1].axis('off')
    ax[1].set_title('encoded')
    plt.show()


def train_gan(model, img, lbl, optimizer, criterion):

    batch_size = img.shape[0]
    # print(img.shape)

    # Zero the parameter's gradient
    optimizer[0].zero_grad()  # 0 = generator
    optimizer[1].zero_grad()  # 1 = discriminator
    img = img.cuda()

    ###################################################################
    # | Discriminator training | Loss: log(D(x)) + log(1 - D(G(z))) | #
    ###################################################################
    model.discriminator.zero_grad()

    #############
    # log(D(x)) #
    #############

    real_labels = torch.full((batch_size,), real_label, dtype=torch.long, device='cuda:0')
    out = model.discriminator(img)
    disc_real_loss = criterion(out, real_labels)
    disc_real_loss.backward()

    ####################
    # log(1 - D(G(z))) #
    ####################

    latent_vector = torch.randn(batch_size, latent_dim, 4, 4, device='cuda:0')
    # print(latent_vector.shape)
    fake_img = model.backbone(latent_vector)
    # print(fake_img.shape)
    fake_labels = torch.full((batch_size,), fake_label, dtype=torch.long, device='cuda:0')

    # Detaching fake images makes it so that gradients for the generator are not calculated
    out = model.discriminator(fake_img.detach())
    disc_fake_loss = criterion(out, fake_labels)
    disc_fake_loss.backward()

    disc_loss = disc_real_loss + disc_fake_loss
    optimizer[1].step()

    ###############################################
    # | Generator training | Loss: log(D(G(z))) | #
    ###############################################
    model.backbone.zero_grad()
    fake_img = model.backbone(latent_vector)
    misleading_labels = torch.full((batch_size,), real_label, dtype=torch.long, device='cuda:0')
    out = model.discriminator(fake_img)
    gen_loss = criterion(out, misleading_labels)
    gen_loss.backward()

    optimizer[0].step()

    # temp variable needed to only visualize the first image of the validation loop
    global visualized
    visualized = False

    return disc_loss.item(), torch.argmax(out, dim=1).sum().item()


def val_gan(model, img, lbl, criterion):
    model.eval()

    img = img.cuda()

    with torch.no_grad():

        batch_size = img.shape[0]

        # Zero the parameter's gradient
        img = img.cuda()

        ###################################################################
        # | Discriminator training | Loss: log(D(x)) + log(1 - D(G(z))) | #
        ###################################################################
        model.discriminator.zero_grad()

        #############
        # log(D(x)) #
        #############

        real_labels = torch.full((batch_size,), real_label, dtype=torch.long, device='cuda:0')
        out = model.discriminator(img)
        disc_real_loss = criterion(out, real_labels)
        ####################
        # log(1 - D(G(z))) #
        ####################

        latent_vector = torch.randn(batch_size, latent_dim, 4, 4, device='cuda:0')
        fake_img = model.backbone(latent_vector)
        fake_labels = torch.full((batch_size,), fake_label, dtype=torch.long, device='cuda:0')
        out = model.discriminator(fake_img.detach())
        disc_fake_loss = criterion(out, fake_labels)

        disc_loss = disc_real_loss + disc_fake_loss

        ###############################################
        # | Generator training | Loss: log(D(G(z))) | #
        ###############################################
        model.backbone.zero_grad()
        misleading_labels = torch.full((batch_size,), real_label, dtype=torch.long, device='cuda:0')
        out = model.discriminator(fake_img)
        gen_loss = criterion(out, misleading_labels)

    global visualized
    if not visualized:
        visualized = True

        visualize(img, fake_img)

    return disc_loss.item(), torch.argmax(out, dim=1).sum().item()
